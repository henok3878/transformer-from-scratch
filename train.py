from functools import partial
import math
import sacrebleu
import os 
import shutil
import wandb 
from typing import cast
import torch 
import argparse 
import torch.nn as nn 
from datetime import datetime 
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler 
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.distributed as dist 
from datasets import Dataset, load_from_disk 
from tokenizers import Tokenizer 


from transformer.config import AppConfig, load_config, TokenizationStrategy 
from transformer.transformer import Transformer
from transformer.components.decoding import greedy_search 

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
    B = len(batch) 
    src_padded = torch.full((B, src_max_len), padding_idx, dtype=torch.long)
    tgt_padded = torch.full((B, tgt_max_len), padding_idx, dtype=torch.long)
    for i, item in enumerate(batch):
        src_seq = torch.tensor(item['src_ids'][:src_max_len], dtype=torch.long)
        tgt_seq = torch.tensor(item['tgt_ids'][:tgt_max_len], dtype=torch.long)

        src_padded[i, :src_seq.size(0)] = src_seq
        tgt_padded[i, :tgt_seq.size(0)] = tgt_seq

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
            name = os.path.basename(p=run_path)
            wandb.init(
                project="transformer-from-scratch",
                name=name,
                config=config.model_dump(),
                id=name, 
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
            self.model.parameters(), lr=self.config.training.lr_factor, betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),weight_decay=self.config.training.weight_decay, eps=self.config.training.adam_eps, 
            fused=True 
        )
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self._noam_lambda)

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing, ignore_index=self.padding_idx) 

        self.train_loader = self._prepare_dataloader(self.train_dataset, shuffle=True) 
        self.full_val_loader = self._prepare_dataloader(dataset=self.full_val_dataset, shuffle=False)
        self.quick_val_loader = self._prepare_dataloader(self.quick_val_dataset, shuffle=False)
        
        self.global_step = 0 
        self.current_epoch = 0 
        # the index of the epoch to start when (re)enter training 
        self.epochs_run = 0  
        self.last_full_val_step = -1 
        self._load_checkpoint()

    def _noam_lambda(self, step: int):
        step += 1  
        d_model = self.config.model.d_model
        warmup_steps = self.config.training.warmup_steps
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)                    

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
        self.global_step = checkpoint["global_step"]
        self.epochs_run = checkpoint["epoch"] + 1
        self.current_epoch = self.epochs_run 
        if self.global_rank == 0:
            print(f"Resumed training from epoch {self.epochs_run}")
    
    def _save_checkpoint(self, value: int, is_epoch: bool):
        """Save checkpoint with cleanup of old checkpoints."""
        if self.global_rank != 0:
            return

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        prefix = "epoch" if is_epoch else "step"
        # save current checkpoint
        ckp_path = os.path.join(self.checkpoint_dir, f"checkpoint_{prefix}_{value}.pt")
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        
        torch.save(checkpoint, ckp_path)
        torch.save(checkpoint, latest_path)
        
        if not is_epoch:
            # clean up old checkpoints
            self._cleanup_old_checkpoints()

        # log checkpoints as wandb artifact 
        artifact = wandb.Artifact("checkpoint", type="model")
        artifact.add_file(ckp_path)
        wandb.log_artifact(artifact)
        
    def _cleanup_old_checkpoints(self):
        """Keep only the last N best step checkpoints.""" 
        if self.global_rank != 0:
            return
            
        checkpoint_files = []
        for file in os.listdir(self.checkpoint_dir):
            if file.startswith("checkpoint_step_") and file.endswith(".pt"):
                try:
                    step_num = int(file.split("_")[2].split(".")[0])
                    checkpoint_files.append((step_num, file))
                except Exception:
                    continue
        
        # sort by step and keep only the last N
        checkpoint_files.sort(key=lambda x: x[0])
        if len(checkpoint_files) > self.config.experiment.keep_last_n:
            files_to_remove = checkpoint_files[:-self.config.experiment.keep_last_n]
            for _, filename in files_to_remove:
                os.remove(os.path.join(self.checkpoint_dir, filename))
                print(f"Removed old checkpoint: {filename}")

    @torch.inference_mode() 
    def _distributed_compute_bleu(self, loader: DataLoader) -> float:
        self.model.eval() 
        local_hyps, local_refs = [], []

        for batch in loader:
            # move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # greedy decode
            preds = greedy_search(
                model=self.model,
                src_ids=batch["src_ids"],
                src_mask=batch["src_mask"],
                tokenizer_tgt=self.tokenizer_tgt,
                max_len=self.config.model.tgt_max_len
            )  

            # detokenise
            local_hyps.extend(self.tokenizer_tgt.decode_batch(preds, skip_special_tokens=True))

            # references already tokenised, so detokenise once
            local_refs.extend(self.tokenizer_tgt.decode_batch(
                batch["label"].tolist(), skip_special_tokens=True))
        
        world_size = self.world_size 
        gathered_hyps: list[list[str]] = [[] for _ in range(world_size)]
        gathered_refs: list[list[str]] = [[] for _ in range(world_size)]

        dist.all_gather_object(gathered_hyps, local_hyps)
        dist.all_gather_object(gathered_refs, local_refs)
        
        if self.global_rank == 0:
            hyps = [h for sublist in gathered_hyps for h in sublist]
            refs = [r for sublist in gathered_refs for r in sublist]
            bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a", lowercase=False)
            score = bleu.score
        else:
            score = 0.0

        self.model.train()
        return score 
    

    @torch.inference_mode()
    def _eval_loader(self, loader: DataLoader): 
        self.model.eval() 
        total_loss = torch.tensor(0.0, device=self.device)
        total_tokens = torch.tensor(0.0, device=self.device)
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            logits = self.model(
                src_ids=batch["src_ids"],
                tgt_ids=batch["tgt_ids"],
                src_mask=batch["src_mask"],
                tgt_mask=batch["tgt_mask"],
                kv_mask=batch["src_mask"]
            )
            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), batch["label"].reshape(-1))
            valid_tokens = (batch["label"] != self.padding_idx).sum() 
            total_loss += loss * valid_tokens 
            total_tokens += valid_tokens 
        # aggregate across processes
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

        avg_loss = (total_loss / total_tokens).item()
        ppl = math.exp(avg_loss)

        self.model.train() 
        return avg_loss, ppl
    
    def _run_full_validation(self):
        avg_loss, ppl = self._eval_loader(loader=self.full_val_loader) 
        train_bleu = self._distributed_compute_bleu(self.quick_val_loader)
        if self.global_rank == 0:
            wandb.log({"val/full_loss": avg_loss, "val/full_ppl": ppl, "val/bleu": train_bleu, "step": self.global_step})
            print(f"[FULL VAL] epoch={self.current_epoch} step={self.global_step} loss={avg_loss:.4f} ppl={ppl:.2f} bleu={train_bleu:.2f}")    
        self.last_full_val_step = self.global_step 
        return avg_loss, ppl 
        

    def _run_quick_validation(self): 
        avg_loss, ppl = self._eval_loader(loader=self.quick_val_loader) 
        if self.global_rank == 0:
            wandb.log({"val/quick_loss": avg_loss, "val/quick_ppl": ppl,  "step": self.global_step})
            print(f"[QUICK VAL] epoch={self.current_epoch} step={self.global_step} loss={avg_loss:.4f} ppl={ppl:.2f}")
        return avg_loss, ppl 

    def _run_batch(self, batch: dict[str, torch.Tensor]) -> float:
        """Runs a single training step."""
        self.global_step += 1

        for key, value in batch.items():
            batch[key] = value.to(self.device)
        
        logits = self.model(
            src_ids=batch["src_ids"],
            tgt_ids=batch["tgt_ids"],
            src_mask=batch["src_mask"],
            tgt_mask=batch["tgt_mask"], 
            kv_mask=batch["src_mask"]
        )

        loss = self.loss_fn(
            logits.reshape(-1, logits.size(-1)), 
            batch["label"].reshape(-1)
        )
            
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.training.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step() 

        # quick & full eval
        if self.global_step % self.config.training.quick_eval_every == 0:
            self._run_quick_validation()
        if self.global_step % self.config.training.full_eval_every == 0:
            self._run_full_validation()

        if self.global_rank == 0:
            # logging
            if self.global_step % self.config.experiment.log_every == 0:
                wandb.log({"train/loss": loss.item(), "step": self.global_step})


            # checkpoint
            if self.global_step % self.config.experiment.save_every_steps == 0:
                self._save_checkpoint(self.global_step, is_epoch=False)

        return loss.item()

    def _run_epoch(self, epoch: int):
        """Run a single training epoch."""
        sampler = cast(DistributedSampler, self.train_loader.sampler)
        sampler.set_epoch(epoch)
        
        self.model.train()
        for batch in self.train_loader:
            self._run_batch(batch)

    def train(self):
        """Main training loop."""
        for epoch in range(self.epochs_run, self.config.training.epochs):
            self.current_epoch = epoch 
            self._run_epoch(epoch)
            if self.global_rank == 0:
                self._save_checkpoint(epoch, is_epoch=True)

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