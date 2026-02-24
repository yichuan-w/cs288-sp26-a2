#!/usr/bin/env python3
"""
Evaluate LC checkpoints (no DDP, single GPU, low memory for prompting).

Usage:
  CUDA_VISIBLE_DEVICES=0 python part4/eval_lc_ckpt.py
"""
import sys
from pathlib import Path

_user_site = Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if _user_site.exists() and str(_user_site) not in sys.path:
    sys.path.insert(0, str(_user_site))

import json
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from part1.train_bpe import train_bpe
from part1.tokenizer import get_tokenizer
from part2.model import TransformerLM
from part4.datasets import create_qa_dataloader
from part4.qa_model import TransformerForMultipleChoice, evaluate_qa_model
from part4.prompting import PromptTemplate, PromptingPipeline, evaluate_prompting as eval_prompting

CONFIG = {
    "pretrain_data": Path(__file__).parent / "fixtures/tinystories_100k.txt",
    "qa_dev": Path(__file__).parent / "fixtures/squad_dev.json",
    "qa_train": Path(__file__).parent / "fixtures/squad_train.json",
    "vocab_size": 8192,
    "d_model": 512,
    "num_layers": 8,
    "num_heads": 8,
    "d_ff": 2048,
    "context_length": 1536,
    "qa_batch_size": 4,
    "max_prompt_tokens": 1536,
    "few_shot_k": 3,
}
CKPT_DIR = Path(__file__).parent / "outputs"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = dict(CONFIG)

    fallback_pt = Path(__file__).parent.parent / "part1/fixtures/tinystories_sample_5M.txt"
    fallback_dev = Path(__file__).parent / "fixtures/qa_dev.json"
    fallback_train = Path(__file__).parent / "fixtures/qa_train.json"
    if not config["pretrain_data"].exists() and fallback_pt.exists():
        config["pretrain_data"] = fallback_pt
    if not config["qa_dev"].exists() and fallback_dev.exists():
        config["qa_dev"] = fallback_dev
    if not config["qa_train"].exists() and fallback_train.exists():
        config["qa_train"] = fallback_train

    if not config["pretrain_data"].exists():
        print(f"Dataset not found: {config['pretrain_data']}")
        return
    if not config["qa_dev"].exists():
        print(f"QA dev not found: {config['qa_dev']}")
        return
    if not config["qa_train"].exists():
        print(f"QA train not found: {config['qa_train']}")
        return

    ckpt_path = CKPT_DIR / "qa_finetuned_latest.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    print("=" * 60)
    print("LC Checkpoint Evaluation (single GPU)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    # Step 1: Train tokenizer (must match training)
    print("\n" + "=" * 60)
    print("STEP 1: Training BPE Tokenizer")
    print("=" * 60)
    special_tokens = ["<|endoftext|>", "<|pad|>"]
    vocab, merges = train_bpe(
        input_path=config["pretrain_data"],
        vocab_size=config["vocab_size"],
        special_tokens=special_tokens,
    )
    tokenizer = get_tokenizer(vocab, merges, special_tokens)
    print(f"Vocab: {len(vocab)}, Merges: {len(merges)}")

    # Step 2: Build model and load checkpoint
    print("\n" + "=" * 60)
    print("STEP 2: Loading checkpoint")
    print("=" * 60)
    transformer = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    )
    qa_model = TransformerForMultipleChoice(
        transformer_lm=transformer,
        hidden_size=config["d_model"],
        num_choices=4,
        pooling="last",
        freeze_backbone=False,
    )
    state = torch.load(ckpt_path, map_location=device)
    qa_model.load_state_dict(state, strict=True)
    qa_model = qa_model.to(device)
    qa_model.eval()
    print(f"Loaded {sum(p.numel() for p in qa_model.parameters()):,} params")

    # Step 3: Prompting evaluation (low memory: max 512 tokens)
    print("\n" + "=" * 60)
    print(f"STEP 3: Evaluating Prompting (max {config['max_prompt_tokens']} tokens)")
    print("=" * 60)
    with open(config["qa_dev"], encoding="utf-8") as f:
        dev_data = json.load(f)
    with open(config["qa_train"], encoding="utf-8") as f:
        train_data = json.load(f)
    exemplars = []
    for ex in train_data[: config["few_shot_k"]]:
        answer_idx = ex.get("answer", -1)
        if 0 <= answer_idx < len(ex["choices"]):
            exemplars.append({
                "context": ex["context"],
                "question": ex["question"],
                "answer_text": ex["choices"][answer_idx],
            })
    template = PromptTemplate(template_name="simple")
    pipeline = PromptingPipeline(
        model=qa_model.transformer,
        tokenizer=tokenizer,
        template=template,
        device=str(device),
        max_prompt_tokens=config["max_prompt_tokens"],
        exemplars=exemplars,
    )
    prompting_res = eval_prompting(pipeline, dev_data)
    print(f"Prompting accuracy: {prompting_res['accuracy']:.2%}")

    # Step 4: Fine-tuned evaluation
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating Fine-tuned Model")
    print("=" * 60)
    dl = create_qa_dataloader(
        dev_data, tokenizer,
        batch_size=config["qa_batch_size"],
        max_length=config["context_length"],
        shuffle=False,
    )
    finetuned_res = evaluate_qa_model(qa_model, dl, str(device))
    print(f"Fine-tuned accuracy: {finetuned_res['accuracy']:.2%}")

    # Save results
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Prompting:   {prompting_res['accuracy']:.2%}")
    print(f"Fine-tuned: {finetuned_res['accuracy']:.2%}")

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "finetuned_predictions.json", "w", encoding="utf-8") as f:
        json.dump({
            "predictions": finetuned_res.get("predictions", []),
            "accuracy": finetuned_res["accuracy"],
            "config": "lc",
        }, f, indent=2)
    with open(out_dir / "prompting_predictions.json", "w", encoding="utf-8") as f:
        json.dump({
            "predictions": prompting_res.get("predictions", []),
            "accuracy": prompting_res["accuracy"],
            "config": "lc",
        }, f, indent=2)
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
