from functools import partial
import os 
import shutil
from typing import cast
import torch 
import argparse 
import torch.nn as nn 
from datetime import datetime 
from torch.utils.data import DataLoader, DistributedSampler 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group, destroy_process_group, broadcast_object_list
from datasets import load_from_disk 
from tokenizers import Tokenizer 

from config import AppConfig, load_config, TokenizationStrategy 
from transformer.transformer import Transformer 

def ddp_setup():
    """initializes the distributed process group"""
    local_rank = int(os.environ["LOCAL_RANK"]) 
    torch.cuda.set_device(local_rank)
    init_process_group(backend='nccl', init_method="env://") 

def ddp_cleanup():
    """clean up distributed process group"""
    destroy_process_group()

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

def prepare_batch(batch: list[dict[str, list[int]]], padding_idx: int) -> dict[str, torch.Tensor]:
    src_ids = [torch.tensor(item['src_ids'], dtype=torch.long) for item in batch]
    tgt_ids = [torch.tensor(item['tgt_ids'], dtype=torch.long) for item in batch]

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
        self.device = torch.device("cuda", self.local_rank) 
        self.config = config 
        self.run_path = run_path
        self.padding_idx = 0 
        
        # create experiment directories
        self.checkpoint_dir = os.path.join(run_path, config.experiment.checkpoint_dir)
        self.log_dir = os.path.join(run_path, config.experiment.log_dir)
        
        self._load_tokenizers_and_datasets() 
        
        model = Transformer(
            model_config=self.config.model, 
            data_config=self.config.data 
        ).to(self.device) 

        self.model = DDP(model, device_ids=[self.local_rank])
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.training.lr
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.padding_idx) 

        self.train_loader = self._prepare_dataloader(self.train_dataset) 
        
        self.epochs_run = 0
        self._load_checkpoint()

    def _load_tokenizers_and_datasets(self):
        if self.global_rank == 0:
            print("Loading tokenizers and dataset...")

        strategy = self.config.data.tokenization_strategy
        
        if strategy == TokenizationStrategy.JOINT:
            vocab_size = self.config.model.src_vocab_size
            processed_dir = f"data/processed-joint-{self.config.data.subset}-vocab{vocab_size}"
            self.train_dataset = load_from_disk(processed_dir)
            
            tokenizer_path = f"tokenizers/tokenizer-joint-{self.config.data.subset}-vocab{vocab_size}.json"
            self.tokenizer_src = Tokenizer.from_file(tokenizer_path)
            self.tokenizer_tgt = self.tokenizer_src
            
        elif strategy == TokenizationStrategy.SEPARATE:
            src_vocab = self.config.model.src_vocab_size
            tgt_vocab = self.config.model.tgt_vocab_size
            processed_dir = f"data/processed-separate-{self.config.data.subset}-src{src_vocab}-tgt{tgt_vocab}"
            self.train_dataset = load_from_disk(processed_dir)

            tokenizer_path_src = f"tokenizers/tokenizer-separate-{self.config.data.lang_src}-vocab{src_vocab}.json"
            tokenizer_path_tgt = f"tokenizers/tokenizer-separate-{self.config.data.lang_tgt}-vocab{tgt_vocab}.json"
            self.tokenizer_src = Tokenizer.from_file(tokenizer_path_src)
            self.tokenizer_tgt = Tokenizer.from_file(tokenizer_path_tgt)
        else:
            raise ValueError(f"Invalid tokenization strategy: {strategy}")

        self.padding_idx = self.tokenizer_src.token_to_id("[PAD]")
    
    def _prepare_dataloader(self, dataset, is_train=True):
        collate = partial(prepare_batch, padding_idx=self.padding_idx)
        return DataLoader(
            dataset, 
            batch_size=self.config.training.batch_size, 
            sampler=DistributedSampler(dataset, shuffle=is_train), 
            collate_fn=collate,
            pin_memory=True, 
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
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        torch.save(checkpoint, ckp_path)
        torch.save(checkpoint, latest_path)
        
        # clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        print(f"Checkpoint saved to {ckp_path}")
    
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
    
    def _should_save_checkpoint(self, epoch: int) -> bool:
        """Check if we should save a checkpoint this epoch."""
        return epoch % self.config.experiment.save_every_epoch == 0
    
    def _run_batch(self, batch: dict[str, torch.Tensor]) -> float:
        """Runs a single training step."""
        for key, value in batch.items():
            batch[key] = value.to(self.device)
        
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
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def _run_epoch(self, epoch: int):
        """Run a single training epoch."""
        sampler = cast(DistributedSampler, self.train_loader.sampler)
        sampler.set_epoch(epoch)
        
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            loss = self._run_batch(batch)
            
            if self.global_rank == 0 and i % self.config.experiment.log_every == 0:
                print(f"Epoch {epoch} | Batch {i}/{len(self.train_loader)} | Loss: {loss:.4f}")

    def train(self):
        """Main training loop."""
        for epoch in range(self.epochs_run, self.config.training.epochs):
            self._run_epoch(epoch)
            
            if self._should_save_checkpoint(epoch):
                self._save_checkpoint(epoch)

def main():
    parser = argparse.ArgumentParser(description="Transformer Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
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
    broadcast_object_list(object_list, src=0)
    run_path = object_list[0]

    assert isinstance(run_path, str), "run_path should be a string"
    try:
        trainer = Trainer(config, run_path)
        trainer.train()
    finally:
        ddp_cleanup()

if __name__ == "__main__":
    main()