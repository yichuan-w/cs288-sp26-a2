"""
Training utilities.
Example submission.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import time
import sys

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import cross_entropy, gradient_clipping


@dataclass
class TrainingConfig:
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    batch_size: int = 8
    log_interval: int = 10
    save_interval: int = 500
    checkpoint_dir: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = False
    patience: Optional[int] = None
    # DDP: rank 0 logs/saves; None = single process
    ddp_rank: Optional[int] = None


class Trainer:
    def __init__(self, model: nn.Module, config: TrainingConfig, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None, compute_loss_fn: Optional[Callable] = None):
        self.model = model.to(config.device)
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.compute_loss_fn = compute_loss_fn or self._default_lm_loss
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        total_steps = len(train_dataloader) * config.num_epochs
        if config.warmup_steps > 0:
            warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=config.warmup_steps)
            main = CosineAnnealingLR(self.optimizer, T_max=total_steps - config.warmup_steps)
            self.scheduler = SequentialLR(self.optimizer, [warmup, main], milestones=[config.warmup_steps])
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []
    
    def _default_lm_loss(self, batch: Dict[str, torch.Tensor], model: nn.Module) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.config.device)
        labels = batch["labels"].to(self.config.device)
        logits = model(input_ids)
        batch_size, seq_len, vocab_size = logits.shape
        return cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
    
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in self.train_dataloader:
            self.optimizer.zero_grad()
            loss = self.compute_loss_fn(batch, self.model)
            loss.backward()
            gradient_clipping(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    @torch.no_grad()
    def evaluate(self) -> float:
        if self.val_dataloader is None:
            return 0.0
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        for batch in self.val_dataloader:
            loss = self.compute_loss_fn(batch, self.model)
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, epoch_callback=None) -> Dict[str, Any]:
        """Train for num_epochs. epoch_callback(epoch, train_loss, val_loss) called each epoch."""
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            val_loss = 0.0
            if self.val_dataloader:
                val_loss = self.evaluate()
                self.val_losses.append(val_loss)
            if epoch_callback:
                epoch_callback(epoch + 1, train_loss, val_loss)
        return {"train_losses": self.train_losses, "val_losses": self.val_losses}


def compute_qa_loss(batch: Dict[str, torch.Tensor], model: nn.Module, device: str = "cuda") -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids, attention_mask)
    return cross_entropy(logits, labels)


def create_qa_loss_fn(device: str = "cuda") -> Callable:
    return lambda batch, model: compute_qa_loss(batch, model, device)


def unwrap_ddp(model: nn.Module) -> nn.Module:
    """Get the underlying model when wrapped in DistributedDataParallel."""
    if hasattr(model, "module"):
        return model.module
    return model
