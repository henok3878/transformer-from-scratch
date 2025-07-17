from functools import partial
import math
import os 
import shutil
import wandb 
from typing import cast
import torch 
from torch import autocast, GradScaler
import argparse 
import torch.nn as nn 
from datetime import datetime 
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler 
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.distributed as dist 
from datasets import Dataset, load_from_disk 
from tokenizers import Tokenizer 

from config import AppConfig, load_config, TokenizationStrategy 
from transformer.transformer import Transformer 

def ddp_setup():
    """initializes the distributed process group"""
    local_rank = int(os.environ["LOCAL_RANK"]) 
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method="env://") 

def ddp_cleanup():
    """clean up distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()

def create_masks(src_batch: torch.Tensor, tgt_batch: torch.Tensor, padding_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        - src_batch: shape = (batch_size, src_seq_len) 
        - tgt_batch: shape = (batch_size, tgt_seq_len) 
        - padding_idx: padding token 
    """
    # (batch_size,src_seq_len) -> (batch_size,1, 1, src_seq_len)
    src_mask = (src_batch != padding_idx).unsqueeze(dim=1).unsqueeze(2)  
    # (batch_size, tgt_seq_len) -> (batch_size, 1, 1, tgt_seq_len)
    tgt_padding_mask = (tgt_batch != padding_idx).unsqueeze(dim=1).unsqueeze(dim=2) 

    tgt_len = tgt_batch.size(1) 
    tgt_lower_matrix = torch.tril(torch.ones(size=(tgt_len, tgt_len),device=tgt_batch.device)).bool() 
    
    # (batch_size, 1, 1, tgt_seq_len) & (tgt_len, tgt_len) -> (batch_size, 1, tgt_len, tgt_len)
    tgt_mask = tgt_padding_mask & tgt_lower_matrix 
    return src_mask, tgt_mask 

def prepare_batch(batch: list[dict[str, list[int]]], padding_idx: int, src_max_len: int, tgt_max_len: int) -> dict[str, torch.Tensor]:
    src_ids = [torch.tensor(item['src_ids'][:src_max_len], dtype=torch.long) for item in batch]
    tgt_ids = [torch.tensor(item['tgt_ids'][:tgt_max_len], dtype=torch.long) for item in batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_ids, batch_first=True, padding_value=padding_idx) 
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_ids,batch_first=True, padding_value=padding_idx)
    
    # the input without of the last token 
    decoder_input = tgt_padded[:, :-1] 
    # input shifted to the right by one 
    label = tgt_padded[:,1:] 

    src_mask, tgt_mask = create_masks(src_padded, decoder_input, padding_idx) 
        
    return {
        "src_ids": src_padded, 
        "tgt_ids": decoder_input, 
        "src_mask": src_mask, 
        "tgt_mask": tgt_mask, 
        "label": label
    }

class Trainer:
    def __init__(self, config: AppConfig, run_path: str):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.world_size = dist.get_world_size()
        self.device = torch.device("cuda", self.local_rank) 
        self.config = config 
        self.run_path = run_path
        self.padding_idx = 0 
        
        # create experiment directories
        self.checkpoint_dir = os.path.join(run_path, config.experiment.checkpoint_dir)
        self.log_dir = os.path.join(run_path, config.experiment.log_dir)

        if self.global_rank == 0:
            wandb.init(
                project="transformer-from-scratch",
                name=os.path.basename(run_path),
                config=config.model_dump(),
                dir=self.log_dir,
                resume="allow"
            )
        
        self._load_tokenizers_and_datasets() 
        
        torch.set_float32_matmul_precision("high")
        model = Transformer(
            model_config=self.config.model, 
            data_config=self.config.data 
        ).to(self.device) 
        model = torch.compile(model=model) 

        self.model = DDP(model, device_ids=[self.local_rank])
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.training.peak_lr, weight_decay=config.training.weight_decay, eps=config.training.adam_eps, 
            fused=True 
        )
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self._lr_lambda)

        self.scaler = GradScaler() 

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing, ignore_index=self.padding_idx) 

        self.train_loader = self._prepare_dataloader(self.train_dataset, shuffle=True) 
        self.full_val_loader = self._prepare_dataloader(dataset=self.full_val_dataset, shuffle=False)
        self.quick_val_loader = self._prepare_dataloader(self.quick_val_dataset, shuffle=False)
        
        self.global_step = 0 
        self.epochs_run = 0 
        self._load_checkpoint()

    def _lr_lambda(self, step: int):
        warmup = self.config.training.warmup_steps
        if step < warmup:
            return step / warmup 
        return (step / warmup) ** -0.5 

    def _load_tokenizers_and_datasets(self):
        if self.global_rank == 0:
            print("Loading tokenizers and dataset...")

        strategy = self.config.data.tokenization_strategy
        
        if strategy == TokenizationStrategy.JOINT:
            vocab_size = self.config.model.src_vocab_size
            processed_dir = f"data/processed-joint-{self.config.data.subset}-vocab{vocab_size}"
            tokenizer_path = f"tokenizers/tokenizer-joint-{self.config.data.subset}-vocab{vocab_size}.json"
            self.tokenizer_src = Tokenizer.from_file(tokenizer_path)
            self.tokenizer_tgt = self.tokenizer_src
            
        else: # SEPARATE 
            src_vocab = self.config.model.src_vocab_size
            tgt_vocab = self.config.model.tgt_vocab_size
            processed_dir = f"data/processed-separate-{self.config.data.subset}-src{src_vocab}-tgt{tgt_vocab}"
            tokenizer_path_src = f"tokenizers/tokenizer-separate-{self.config.data.lang_src}-vocab{src_vocab}.json"
            tokenizer_path_tgt = f"tokenizers/tokenizer-separate-{self.config.data.lang_tgt}-vocab{tgt_vocab}.json"
            self.tokenizer_src = Tokenizer.from_file(tokenizer_path_src)
            self.tokenizer_tgt = Tokenizer.from_file(tokenizer_path_tgt)

        train_dir = os.path.join(processed_dir, "train")
        val_dir = os.path.join(processed_dir, "validation")

        self.train_dataset: Dataset = load_from_disk(train_dir)  # type: ignore 
        self.full_val_dataset: Dataset = load_from_disk(val_dir) # type: ignore 

        quick_n = min(len(self.full_val_dataset), self.config.training.quick_val_size) 
        self.quick_val_dataset = self.full_val_dataset.select(range(quick_n)) 

        self.padding_idx = self.tokenizer_src.token_to_id("[PAD]")
    
    def _prepare_dataloader(self, dataset, shuffle: bool):
        collate = partial(prepare_batch, padding_idx=self.padding_idx,
                          src_max_len=self.config.model.src_max_len,
                          tgt_max_len=self.config.model.tgt_max_len)
        return DataLoader(
            dataset, 
            batch_size=self.config.training.batch_size, 
            sampler=DistributedSampler(dataset, shuffle=shuffle, drop_last=shuffle), # only drop the final short batch in training 
            drop_last=shuffle,
            collate_fn=collate,
            pin_memory=True, 
            pin_memory_device="cuda",  
            num_workers=self.config.training.num_workers
        ) 
    
    def _load_checkpoint(self):
        """Load the latest checkpoint if it exists."""
        ckp_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        if not os.path.exists(ckp_path):
            if self.global_rank == 0:
                print("No checkpoint found, starting from scratch.")
            return
            
        map_location = {'cuda:0': f'cuda:{self.local_rank}'}
        checkpoint = torch.load(ckp_path, map_location=map_location)
        self.model.module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(state_dict=checkpoint['scheduler_state_dict']) 
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epochs_run = checkpoint["epoch"] + 1
        if self.global_rank == 0:
            print(f"Resumed training from epoch {self.epochs_run}")
    
    def _save_checkpoint(self, epoch: int):
        """Save checkpoint with cleanup of old checkpoints."""
        if self.global_rank != 0:
            return
            
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # save current checkpoint
        ckp_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }
        
        torch.save(checkpoint, ckp_path)
        torch.save(checkpoint, latest_path)
        
        # clean up old checkpoints
        self._cleanup_old_checkpoints()

        # log checkpoints as wandb artifact 
        artifact = wandb.Artifact("checkpoint", type="model")
        artifact.add_file(ckp_path)
        wandb.log_artifact(artifact)
        
    def _cleanup_old_checkpoints(self):
        """Keep only the last N checkpoints."""
        if self.global_rank != 0:
            return
            
        checkpoint_files = []
        for file in os.listdir(self.checkpoint_dir):
            if file.startswith("checkpoint_epoch_") and file.endswith(".pt"):
                epoch_num = int(file.split("_")[2].split(".")[0])
                checkpoint_files.append((epoch_num, file))
        
        # sort by epoch and keep only the last N
        checkpoint_files.sort(key=lambda x: x[0])
        if len(checkpoint_files) > self.config.experiment.keep_last_n:
            files_to_remove = checkpoint_files[:-self.config.experiment.keep_last_n]
            for _, filename in files_to_remove:
                os.remove(os.path.join(self.checkpoint_dir, filename))
                print(f"Removed old checkpoint: {filename}")
    
    @torch.inference_mode()
    def _eval_loader(self, loader: DataLoader): 
        total_loss = torch.tensor(0.0, device=self.device)
        total_tokens = torch.tensor(0.0, device=self.device)
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with autocast(device_type="cuda"):
                logits = self.model(
                    src_ids=batch["src_ids"],
                    tgt_ids=batch["tgt_ids"],
                    src_mask=batch["src_mask"],
                    tgt_mask=batch["tgt_mask"],
                )
                loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), batch["label"].reshape(-1))
            total_loss += loss * batch["label"].numel()
            total_tokens += batch["label"].numel()

        # aggregate across processes
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

        avg_loss = (total_loss / total_tokens).item()
        ppl = math.exp(avg_loss)
        return avg_loss, ppl
    
    def _run_full_validation(self):
        avg_loss, ppl = self._eval_loader(loader=self.full_val_loader) 
        if self.global_rank == 0:
            wandb.log({"val/full_loss": avg_loss, "val/full_ppl": ppl, "step": self.global_step})
            print(f"[FULL VAL] step={self.global_step} loss={avg_loss:.4f} ppl={ppl:.2f}")    
        

    def _run_quick_validation(self): 
        avg_loss, ppl = self._eval_loader(loader=self.quick_val_loader) 
        if self.global_rank == 0:
            wandb.log({"val/quick_loss": avg_loss, "val/quick_ppl": ppl, "step": self.global_step})
            print(f"[QUICK VAL] step={self.global_step} loss={avg_loss:.4f} ppl={ppl:.2f}")

    def _run_batch(self, batch: dict[str, torch.Tensor]) -> float:
        """Runs a single training step."""
        self.global_step += 1

        for key, value in batch.items():
            batch[key] = value.to(self.device)
        
        with autocast(device_type="cuda"): 
            logits = self.model(
                src_ids=batch["src_ids"],
                tgt_ids=batch["tgt_ids"],
                src_mask=batch["src_mask"],
                tgt_mask=batch["tgt_mask"]
            )

            loss = self.loss_fn(
                logits.reshape(-1, logits.size(-1)), 
                batch["label"].reshape(-1)
            )
            
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer=self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.training.max_grad_norm
        )
        self.scaler.step(optimizer=self.optimizer)
        self.scaler.update()
        self.scheduler.step() 

        if self.global_rank == 0:
            # logging
            if self.global_step % self.config.experiment.log_every == 0:
                wandb.log({"train/loss": loss.item(), "step": self.global_step})

            # quick & full eval
            if self.global_step % self.config.training.quick_eval_every == 0:
                self._run_quick_validation()
            if self.global_step % self.config.training.full_eval_every == 0:
                self._run_full_validation()

            # checkpoint
            if self.global_step % self.config.experiment.save_every_steps == 0:
                self._save_checkpoint(self.global_step)

        return loss.item()

    def _run_epoch(self, epoch: int):
        """Run a single training epoch."""
        sampler = cast(DistributedSampler, self.train_loader.sampler)
        sampler.set_epoch(epoch)
        
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            loss = self._run_batch(batch)

    def train(self):
        """Main training loop."""
        for epoch in range(self.epochs_run, self.config.training.epochs):
            self._run_epoch(epoch)
            if self.global_rank == 0:
                self._run_full_validation()
                self._save_checkpoint(epoch)

def main():
    parser = argparse.ArgumentParser(description="Transformer Training Script")
    parser.add_argument("--config", type=str, default="./configs/config_de-en.yaml", help="Path to config file")
    args = parser.parse_args()
    config = load_config(args.config)

    ddp_setup()

    # create experiment directory
    run_path: str
    if int(os.environ["RANK"]) == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config.data.dataset_name}_{config.data.lang_src}-{config.data.lang_tgt}_{timestamp}"
        run_path = os.path.join(config.experiment.base_dir, run_name)
        print(f"Starting new experiment: {run_path}")
        os.makedirs(run_path, exist_ok=True)
        
        # save config copy for reproducibility
        shutil.copy(args.config, os.path.join(run_path, "config.yaml"))
    else:
        run_path = ""
    
    # broadcast the run_path to all processes
    object_list = [run_path]
    dist.broadcast_object_list(object_list, src=0)
    run_path = object_list[0]

    assert isinstance(run_path, str), "run_path should be a string"
    try:
        trainer = Trainer(config, run_path)
        trainer.train()
    finally:
        if dist.get_rank() == 0:
            wandb.finish()
        ddp_cleanup()

if __name__ == "__main__":
    main()