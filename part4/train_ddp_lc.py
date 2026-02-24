#!/usr/bin/env python3
"""
Part 4 DDP Training - Large Context (1536) to avoid truncation

context_length=1536 covers ~99.4% SQuAD (P99=1460). 1024 would truncate ~8%.
Smaller batch to avoid OOM. Use 4 GPUs (machine has 8 total).

Usage:
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 part4/train_ddp_lc.py
"""
import sys
from pathlib import Path

# Ensure user site-packages (e.g. regex) is findable when using venv without it
_user_site = Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if _user_site.exists() and str(_user_site) not in sys.path:
    sys.path.insert(0, str(_user_site))

import argparse
import json
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from part1.train_bpe import train_bpe
from part1.tokenizer import get_tokenizer
from part2.model import TransformerLM
from part3.nn_utils import cross_entropy, gradient_clipping
from part4.datasets import PretrainingDataset, MultipleChoiceQADataset
from part4.sampling import generate_text
from part4.qa_model import TransformerForMultipleChoice, evaluate_qa_model
from part4.prompting import PromptTemplate, PromptingPipeline, evaluate_prompting as eval_prompting
from part4.trainer import Trainer, TrainingConfig, create_qa_loss_fn, unwrap_ddp


# Large-context config: context_length=1536 covers ~99.4% SQuAD (P99=1460)
# 1024 would truncate ~8% train; 1536 truncates <1%
CONFIG = {
    "pretrain_data": Path(__file__).parent / "fixtures/tinystories_100k.txt",
    "qa_train": Path(__file__).parent / "fixtures/squad_train.json",
    "qa_dev": Path(__file__).parent / "fixtures/squad_dev.json",
    "vocab_size": 8192,
    "d_model": 512,
    "num_layers": 8,
    "num_heads": 8,
    "d_ff": 2048,
    "context_length": 1536,  # 覆盖 99%+ SQuAD，不 truncate
    "pretrain_epochs": 5,
    "finetune_epochs": 15,
    "batch_size": 6,   # 1536 比 1024 多 50% 显存，略减 batch
    "qa_batch_size": 4,
    "lr": 1e-4,
}


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return rank, local_rank, world_size, device
    raise RuntimeError("Use: CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 part4/train_ddp_lc.py")


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def train_tokenizer(pretrain_data: Path, vocab_size: int, rank: int):
    if is_main(rank):
        print("\n" + "=" * 60)
        print("STEP 1: Training BPE Tokenizer")
        print("=" * 60)
    special_tokens = ["<|endoftext|>", "<|pad|>"]
    vocab, merges = train_bpe(input_path=pretrain_data, vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer = get_tokenizer(vocab, merges, special_tokens)
    if is_main(rank):
        print(f"Vocab: {len(vocab)}, Merges: {len(merges)}")
    return tokenizer, vocab, merges


def pretrain_lm(tokenizer, config: dict, device: torch.device, rank: int, world_size: int) -> TransformerLM:
    if is_main(rank):
        print("\n" + "=" * 60)
        print("STEP 2: Pretraining LM (DDP, context=1536)")
        print("=" * 60)

    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    ).to(device)

    dataset = PretrainingDataset(
        config["pretrain_data"], tokenizer,
        max_length=config["context_length"],
        stride=config["context_length"] // 2,
    )
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"],
        sampler=sampler, shuffle=False, num_workers=0, pin_memory=True,
    )

    if is_main(rank):
        print(f"Model: {sum(p.numel() for p in model.parameters()):,} params, context={config['context_length']}")
        print(f"Batches/epoch: {len(dataloader)}")

    model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    train_config = TrainingConfig(
        num_epochs=config["pretrain_epochs"],
        learning_rate=config["lr"],
        weight_decay=0.01,
        warmup_steps=min(100, len(dataloader) // 5),
        max_grad_norm=1.0,
        device=str(device),
    )
    trainer = Trainer(model=model, config=train_config, train_dataloader=dataloader)

    for epoch in range(config["pretrain_epochs"]):
        sampler.set_epoch(epoch)
        loss = trainer.train_epoch()
        trainer.train_losses.append(loss)
        if is_main(rank):
            print(f"  Pretrain epoch {epoch + 1}: loss = {loss:.4f}")

    m = unwrap_ddp(model)
    if is_main(rank):
        print("\nGeneration test:")
        for p in ["Once upon a time", "The little dog"]:
            out = generate_text(m, tokenizer, p, max_new_tokens=30, method="greedy")
            print(f"  '{p}' -> '{out[:60]}...'")
        # Save pretrained LM
        ckpt_dir = Path(__file__).parent / "outputs"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(m.state_dict(), ckpt_dir / "pretrained_lm.pt")
        print(f"  Saved pretrained LM -> {ckpt_dir}/pretrained_lm.pt")

    return m


def finetune_qa(pretrained: TransformerLM, tokenizer, config: dict, device: torch.device, rank: int, world_size: int) -> TransformerForMultipleChoice:
    if is_main(rank):
        print("\n" + "=" * 60)
        print("STEP 3: Fine-tuning QA (DDP, context=1536)")
        print("=" * 60)

    qa_model = TransformerForMultipleChoice(
        transformer_lm=pretrained,
        hidden_size=pretrained.d_model,
        num_choices=4,
        pooling="last",
        freeze_backbone=False,
    ).to(device)

    with open(config["qa_train"]) as f:
        train_data = json.load(f)

    dataset = MultipleChoiceQADataset(train_data, tokenizer, max_length=config["context_length"], num_choices=4)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=config["qa_batch_size"],
        sampler=sampler, shuffle=False, num_workers=0, pin_memory=True,
    )

    if is_main(rank):
        print(f"QA params: {sum(p.numel() for p in qa_model.parameters()):,}")
        print(f"Batches/epoch: {len(dataloader)}")

    qa_model = DDP(qa_model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True)
    train_config = TrainingConfig(
        num_epochs=config["finetune_epochs"],
        learning_rate=config["lr"] / 2,
        weight_decay=0.01,
        warmup_steps=min(50, len(dataloader) // 5),
        max_grad_norm=1.0,
        device=str(device),
    )
    trainer = Trainer(
        model=qa_model,
        config=train_config,
        train_dataloader=dataloader,
        compute_loss_fn=create_qa_loss_fn(str(device)),
    )

    for epoch in range(config["finetune_epochs"]):
        sampler.set_epoch(epoch)
        loss = trainer.train_epoch()
        trainer.train_losses.append(loss)
        if is_main(rank):
            print(f"  Finetune epoch {epoch + 1}: loss = {loss:.4f}")
            # Save checkpoint each epoch (overwrite latest for recovery if killed)
            m = unwrap_ddp(qa_model)
            ckpt_dir = Path(__file__).parent / "outputs"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(m.state_dict(), ckpt_dir / "qa_finetuned_latest.pt")

    return unwrap_ddp(qa_model)


def evaluate_prompting(model, tokenizer, qa_dev_path: Path, device: torch.device, rank: int, max_prompt_tokens: int = 1024) -> dict:
    if not is_main(rank):
        return {"accuracy": 0.0, "predictions": []}
    print("\n" + "=" * 60)
    print(f"STEP 4: Evaluating Prompting (chunk to max {max_prompt_tokens} tokens, avoid OOM)")
    print("=" * 60)
    with open(qa_dev_path) as f:
        dev_data = json.load(f)
    template = PromptTemplate(template_name="simple")
    pipeline = PromptingPipeline(
        model=model,
        tokenizer=tokenizer,
        template=template,
        device=str(device),
        max_prompt_tokens=max_prompt_tokens,
    )
    results = eval_prompting(pipeline, dev_data)
    print(f"Prompting accuracy: {results['accuracy']:.2%}")
    return results


def evaluate_finetuned(qa_model, tokenizer, config: dict, device: torch.device, rank: int) -> dict:
    if not is_main(rank):
        return {"accuracy": 0.0, "predictions": []}
    print("\n" + "=" * 60)
    print("STEP 5: Evaluating Fine-tuned Model")
    print("=" * 60)
    with open(config["qa_dev"]) as f:
        dev_data = json.load(f)
    from part4.datasets import create_qa_dataloader
    dl = create_qa_dataloader(dev_data, tokenizer, batch_size=config["qa_batch_size"], max_length=config["context_length"], shuffle=False)
    results = evaluate_qa_model(qa_model, dl, str(device))
    print(f"Fine-tuned accuracy: {results['accuracy']:.2%}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Part 4 DDP - Large Context (1536)",
        epilog="Run: CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 part4/train_ddp_lc.py"
    )
    parser.add_argument("--test-ckpt", action="store_true", help="Run 1 epoch each to verify checkpoint saving")
    args = parser.parse_args()

    config = dict(CONFIG)

    # Fallback data
    fallback_pt = Path(__file__).parent.parent / "part1/fixtures/tinystories_sample_5M.txt"
    fallback_qa = Path(__file__).parent / "fixtures/qa_train.json"
    fallback_dev = Path(__file__).parent / "fixtures/qa_dev.json"

    if args.test_ckpt:
        config["pretrain_epochs"] = 1
        config["finetune_epochs"] = 1
        config["context_length"] = 512
        config["batch_size"] = 8
        config["qa_batch_size"] = 8
        config["pretrain_data"] = fallback_qa
        config["qa_train"] = fallback_qa
        config["qa_dev"] = fallback_dev

    if not config["pretrain_data"].exists() and fallback_pt.exists():
        config["pretrain_data"] = fallback_pt
    if not config["pretrain_data"].exists():
        print(f"Dataset not found: {config['pretrain_data']}\nRun: python part4/setup_datasets.py")
        return
    if not config["qa_train"].exists() and fallback_qa.exists():
        config["qa_train"] = fallback_qa
        config["qa_dev"] = fallback_dev
    if not config["qa_train"].exists():
        print(f"QA data not found. Run: python part4/setup_datasets.py")
        return

    rank, local_rank, world_size, device = setup_ddp()

    if is_main(rank):
        print("=" * 60)
        print("CS288 Part 4 - DDP Large Context (1536)" + (" [--test-ckpt: 1 epoch each]" if args.test_ckpt else ""))
        print("=" * 60)
        print(f"Context length: {config['context_length']}")
        print(f"World size: {world_size}")
        print(f"Pretrain batch: {config['batch_size']}, QA batch: {config['qa_batch_size']}")

    try:
        tokenizer, _, _ = train_tokenizer(config["pretrain_data"], config["vocab_size"], rank)
        pretrained = pretrain_lm(tokenizer, config, device, rank, world_size)
        qa_model = finetune_qa(pretrained, tokenizer, config, device, rank, world_size)
        prompting_res = evaluate_prompting(
            qa_model.transformer, tokenizer, config["qa_dev"], device, rank,
            max_prompt_tokens=min(1024, config["context_length"]),
        )
        finetuned_res = evaluate_finetuned(qa_model, tokenizer, config, device, rank)

        if is_main(rank):
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Context: {config['context_length']}, Params: {sum(p.numel() for p in pretrained.parameters()):,}")
            print(f"Prompting:   {prompting_res['accuracy']:.2%}")
            print(f"Fine-tuned:  {finetuned_res['accuracy']:.2%}")

            out_dir = Path(__file__).parent / "outputs"
            out_dir.mkdir(exist_ok=True)
            with open(out_dir / "finetuned_predictions.json", "w") as f:
                json.dump({"predictions": finetuned_res.get("predictions", []), "accuracy": finetuned_res["accuracy"], "config": "lc"}, f, indent=2)
            with open(out_dir / "prompting_predictions.json", "w") as f:
                json.dump({"predictions": prompting_res.get("predictions", []), "accuracy": prompting_res["accuracy"], "config": "lc"}, f, indent=2)
            print(f"\nSaved to {out_dir}/")
        if args.test_ckpt:
            ckpt_dir = Path(__file__).parent / "outputs"
            pt_files = list(ckpt_dir.glob("*.pt"))
            print("\n" + "=" * 60)
            print("CHECKPOINT VERIFICATION (--test-ckpt)")
            print("=" * 60)
            for f in pt_files:
                print(f"  OK: {f.name} ({f.stat().st_size:,} bytes)")
            if not pt_files:
                print("  WARNING: No .pt checkpoint files found!")
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
