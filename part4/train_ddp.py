#!/usr/bin/env python3
"""
Part 4 DDP (Distributed Data Parallel) Training Script

Multi-GPU training using PyTorch DistributedDataParallel.
Use torchrun to launch. Example:

  # Use GPUs 1,2,3 (skip GPU 0)
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 part4/train_ddp.py --quick

  # Use all visible GPUs
  torchrun --nproc_per_node=4 part4/train_ddp.py --medium

  # Use 2 GPUs with small config
  torchrun --nproc_per_node=2 part4/train_ddp.py --small
"""

import argparse
import json
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from part1.train_bpe import train_bpe
from part1.tokenizer import get_tokenizer
from part2.model import TransformerLM
from part3.nn_utils import cross_entropy, gradient_clipping
from torch.utils.data import DataLoader
from part4.datasets import PretrainingDataset, MultipleChoiceQADataset, create_qa_dataloader
from part4.sampling import generate_text
from part4.qa_model import TransformerForMultipleChoice, evaluate_qa_model
from part4.prompting import PromptTemplate, PromptingPipeline, evaluate_prompting as eval_prompting
from part4.trainer import Trainer, TrainingConfig, create_qa_loss_fn, unwrap_ddp


# =============================================================================
# Configuration
# =============================================================================

CONFIGS = {
    "quick": {
        "pretrain_data": Path(__file__).parent.parent / "part1/fixtures/tinystories_sample_5M.txt",
        "qa_train": Path(__file__).parent / "fixtures/qa_train.json",
        "qa_dev": Path(__file__).parent / "fixtures/qa_dev.json",
        "vocab_size": 512,
        "d_model": 128,
        "num_layers": 4,
        "num_heads": 4,
        "d_ff": 512,
        "context_length": 256,
        "pretrain_epochs": 3,
        "finetune_epochs": 5,
        "batch_size": 32,
        "lr": 1e-3,
    },
    "small": {
        "pretrain_data": Path(__file__).parent / "fixtures/tinystories_100k.txt",
        "qa_train": Path(__file__).parent / "fixtures/squad_train.json",
        "qa_dev": Path(__file__).parent / "fixtures/squad_dev.json",
        "vocab_size": 4096,
        "d_model": 256,
        "num_layers": 6,
        "num_heads": 8,
        "d_ff": 1024,
        "context_length": 512,
        "pretrain_epochs": 3,
        "finetune_epochs": 10,
        "batch_size": 32,
        "lr": 3e-4,
    },
    "medium": {
        "pretrain_data": Path(__file__).parent / "fixtures/tinystories_100k.txt",
        "qa_train": Path(__file__).parent / "fixtures/squad_train.json",
        "qa_dev": Path(__file__).parent / "fixtures/squad_dev.json",
        "vocab_size": 8192,
        "d_model": 512,
        "num_layers": 8,
        "num_heads": 8,
        "d_ff": 2048,
        "context_length": 512,
        "pretrain_epochs": 5,
        "finetune_epochs": 15,
        "batch_size": 16,
        "lr": 1e-4,
    },
}


def setup_ddp():
    """Initialize distributed process group. Must be called by all processes."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return rank, local_rank, world_size, device
    raise RuntimeError("DDP requires torchrun. Use: torchrun --nproc_per_node=N part4/train_ddp.py ...")


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


# =============================================================================
# Step 1: Train BPE Tokenizer
# =============================================================================

def train_tokenizer(pretrain_data: Path, vocab_size: int, rank: int) -> tuple:
    """Train BPE tokenizer. Run on all ranks (deterministic, same result)."""
    if is_main_process(rank):
        print("\n" + "=" * 60)
        print("STEP 1: Training BPE Tokenizer")
        print("=" * 60)
        print(f"Input: {pretrain_data}")
        print(f"Vocab size: {vocab_size}")

    special_tokens = ["<|endoftext|>", "<|pad|>"]
    vocab, merges = train_bpe(
        input_path=pretrain_data,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    tokenizer = get_tokenizer(vocab, merges, special_tokens)

    if is_main_process(rank):
        test_text = "Once upon a time, there was a little girl."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"\nTokenizer trained! Vocab size: {len(vocab)}, Merges: {len(merges)}")
        print(f"Test: '{test_text}' -> {len(tokens)} tokens -> '{decoded}'")

    return tokenizer, vocab, merges


# =============================================================================
# Step 2: Pretrain LM (DDP)
# =============================================================================

def pretrain_lm(tokenizer, config: dict, device: torch.device, rank: int, world_size: int) -> TransformerLM:
    """Pretrain Transformer LM with DDP."""
    if is_main_process(rank):
        print("\n" + "=" * 60)
        print("STEP 2: Pretraining Language Model (DDP)")
        print("=" * 60)

    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    ).to(device)

    # DistributedSampler for pretraining
    dataset = PretrainingDataset(
        config["pretrain_data"],
        tokenizer,
        max_length=config["context_length"],
        stride=config["context_length"] // 2,
    )
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    if is_main_process(rank):
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel: d_model={config['d_model']}, layers={config['num_layers']}, params={num_params:,}")
        print(f"World size: {world_size}, Batches/epoch: {len(dataloader)}")

    model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    train_config = TrainingConfig(
        num_epochs=config["pretrain_epochs"],
        learning_rate=config["lr"],
        weight_decay=0.01,
        warmup_steps=min(100, len(dataloader) // 5),
        max_grad_norm=1.0,
        device=str(device),
        log_interval=max(1, len(dataloader) // 5),
        ddp_rank=rank,
    )

    trainer = Trainer(
        model=model,
        config=train_config,
        train_dataloader=dataloader,
    )

    for epoch in range(config["pretrain_epochs"]):
        sampler.set_epoch(epoch)
        train_loss = trainer.train_epoch()
        trainer.train_losses.append(train_loss)
        if is_main_process(rank):
            print(f"  Pretrain epoch {epoch + 1}: loss = {train_loss:.4f}")

    # Unwrap DDP for return
    pretrained_model = unwrap_ddp(model)

    if is_main_process(rank):
        print("\nGeneration test:")
        for prompt in ["Once upon a time", "The little dog"]:
            generated = generate_text(
                pretrained_model, tokenizer, prompt,
                max_new_tokens=30,
                method="greedy"
            )
            print(f"  '{prompt}' -> '{generated[:80]}...'")

    return pretrained_model


# =============================================================================
# Step 3: Fine-tune for QA (DDP)
# =============================================================================

def finetune_qa(pretrained_model: TransformerLM, tokenizer, config: dict, device: torch.device, rank: int, world_size: int) -> TransformerForMultipleChoice:
    """Fine-tune for QA with DDP."""
    if is_main_process(rank):
        print("\n" + "=" * 60)
        print("STEP 3: Fine-tuning for Multiple-Choice QA (DDP)")
        print("=" * 60)

    qa_model = TransformerForMultipleChoice(
        transformer_lm=pretrained_model,
        hidden_size=pretrained_model.d_model,
        num_choices=4,
        pooling="last",
        freeze_backbone=False,
    ).to(device)

    with open(config["qa_train"]) as f:
        train_data = json.load(f)

    dataset = MultipleChoiceQADataset(
        train_data, tokenizer,
        max_length=config["context_length"],
        num_choices=4,
    )
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)
    train_dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    if is_main_process(rank):
        print(f"\nQA model params: {sum(p.numel() for p in qa_model.parameters()):,}")
        print(f"Training examples: {len(train_data)}, Batches/epoch: {len(train_dataloader)}")

    # find_unused_parameters=True: QA model may have params not used in every forward (e.g. pooling paths)
    qa_model = DDP(qa_model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True)

    train_config = TrainingConfig(
        num_epochs=config["finetune_epochs"],
        learning_rate=config["lr"] / 2,
        weight_decay=0.01,
        warmup_steps=min(50, len(train_dataloader) // 5),
        max_grad_norm=1.0,
        device=str(device),
        log_interval=max(1, len(train_dataloader) // 5),
        ddp_rank=rank,
    )

    trainer = Trainer(
        model=qa_model,
        config=train_config,
        train_dataloader=train_dataloader,
        compute_loss_fn=create_qa_loss_fn(str(device)),
    )

    for epoch in range(config["finetune_epochs"]):
        sampler.set_epoch(epoch)
        train_loss = trainer.train_epoch()
        trainer.train_losses.append(train_loss)
        if is_main_process(rank):
            print(f"  Finetune epoch {epoch + 1}: loss = {train_loss:.4f}")

    return unwrap_ddp(qa_model)


# =============================================================================
# Step 4 & 5: Evaluate (rank 0 only)
# =============================================================================

def evaluate_prompting(model: TransformerLM, tokenizer, qa_dev_path: Path, device: torch.device, rank: int) -> dict:
    if not is_main_process(rank):
        return {"accuracy": 0.0, "predictions": []}

    print("\n" + "=" * 60)
    print("STEP 4: Evaluating Prompting")
    print("=" * 60)

    with open(qa_dev_path) as f:
        dev_data = json.load(f)

    template = PromptTemplate(template_name="simple")
    pipeline = PromptingPipeline(model=model, tokenizer=tokenizer, template=template, device=str(device))
    results = eval_prompting(pipeline, dev_data)

    print(f"\nPrompting accuracy: {results['accuracy']:.2%}")
    return results


def evaluate_finetuned(qa_model: TransformerForMultipleChoice, tokenizer, config: dict, device: torch.device, rank: int) -> dict:
    if not is_main_process(rank):
        return {"accuracy": 0.0, "predictions": []}

    print("\n" + "=" * 60)
    print("STEP 5: Evaluating Fine-tuned Model")
    print("=" * 60)

    with open(config["qa_dev"]) as f:
        dev_data = json.load(f)

    dev_dataloader = create_qa_dataloader(
        data=dev_data,
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=config["context_length"],
        num_choices=4,
        shuffle=False,
    )

    results = evaluate_qa_model(qa_model, dev_dataloader, str(device))
    print(f"\nFine-tuned accuracy: {results['accuracy']:.2%}")
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Part 4 DDP Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from project root):
  # Use GPUs 4,5,6,7 (后面几个GPU)
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 part4/train_ddp.py --medium

  # Quick test with 2 GPUs
  CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 part4/train_ddp.py --quick

  # Use all visible GPUs
  torchrun --nproc_per_node=4 part4/train_ddp.py --small
        """
    )
    parser.add_argument("--quick", action="store_true", help="Quick test")
    parser.add_argument("--small", action="store_true", help="Small model")
    parser.add_argument("--medium", action="store_true", help="Medium model (default)")
    args = parser.parse_args()

    config_name = "quick" if args.quick else ("small" if args.small else "medium")
    config = dict(CONFIGS[config_name])

    # Fallback: use bundled data when setup_datasets.py hasn't been run
    fallback_pretrain = Path(__file__).parent.parent / "part1/fixtures/tinystories_sample_5M.txt"
    fallback_qa_train = Path(__file__).parent / "fixtures/qa_train.json"
    fallback_qa_dev = Path(__file__).parent / "fixtures/qa_dev.json"

    if not config["pretrain_data"].exists():
        if fallback_pretrain.exists():
            config["pretrain_data"] = fallback_pretrain
        else:
            print(f"Dataset not found: {config['pretrain_data']}")
            print("Run: python part4/setup_datasets.py  or use --quick")
            return
    if not config["qa_train"].exists():
        if fallback_qa_train.exists():
            config["qa_train"] = fallback_qa_train
        else:
            print(f"Dataset not found: {config['qa_train']}")
            print("Run: python part4/setup_datasets.py  or use --quick")
            return
    if not config["qa_dev"].exists():
        if fallback_qa_dev.exists():
            config["qa_dev"] = fallback_qa_dev
        else:
            print(f"Dataset not found: {config['qa_dev']}")
            return

    # Setup DDP
    rank, local_rank, world_size, device = setup_ddp()

    if is_main_process(rank):
        print("=" * 60)
        print("CS288 Part 4 - DDP Training")
        print("=" * 60)
        print(f"\nConfig: {config_name}, World size: {world_size}")
        print(f"Device: {device}")

    try:
        # Step 1: Train tokenizer
        tokenizer, vocab, merges = train_tokenizer(config["pretrain_data"], config["vocab_size"], rank)

        # Step 2: Pretrain LM (DDP)
        pretrained_model = pretrain_lm(tokenizer, config, device, rank, world_size)

        # Step 3: Fine-tune QA (DDP)
        qa_model = finetune_qa(pretrained_model, tokenizer, config, device, rank, world_size)

        # Step 4: Evaluate prompting (rank 0 only)
        prompting_results = evaluate_prompting(qa_model.transformer, tokenizer, config["qa_dev"], device, rank)

        # Step 5: Evaluate fine-tuned (rank 0 only)
        finetuned_results = evaluate_finetuned(qa_model, tokenizer, config, device, rank)

        # Summary and save (rank 0 only)
        if is_main_process(rank):
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Config: {config_name}")
            print(f"Model params: {sum(p.numel() for p in pretrained_model.parameters()):,}")
            print(f"Prompting accuracy:   {prompting_results['accuracy']:.2%}")
            print(f"Fine-tuned accuracy:  {finetuned_results['accuracy']:.2%}")
            print(f"Random baseline:      25.00%")

            prompting_boost = prompting_results["accuracy"] - finetuned_results["accuracy"]
            print(f"\nPrompting boost over fine-tuned: {prompting_boost:+.2%}")

            output_dir = Path(__file__).parent / "outputs"
            output_dir.mkdir(exist_ok=True)

            with open(output_dir / "finetuned_predictions.json", "w") as f:
                json.dump({
                    "predictions": finetuned_results.get("predictions", []),
                    "accuracy": finetuned_results["accuracy"],
                    "config": config_name,
                }, f, indent=2)

            with open(output_dir / "prompting_predictions.json", "w") as f:
                json.dump({
                    "predictions": prompting_results.get("predictions", []),
                    "accuracy": prompting_results["accuracy"],
                    "config": config_name,
                }, f, indent=2)

            print(f"\nPredictions saved to {output_dir}/")
            print("\nDone!")
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
