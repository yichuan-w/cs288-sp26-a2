"""
Prompting utilities for multiple-choice QA.
Example submission.
"""
import torch
from torch import Tensor
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import softmax


class PromptTemplate:
    TEMPLATES = {
        "basic": "Context: {context}\n\nQuestion: {question}\n\nChoices:\n{choices_formatted}\n\nAnswer:",
        "instruction": "Read the following passage and answer the question.\n\nPassage: {context}\n\nQuestion: {question}\n\n{choices_formatted}\n\nSelect the letter:",
        "simple": "{context}\n{question}\n{choices_formatted}\nThe answer is",
    }
    
    def __init__(self, template_name: str = "basic", custom_template: Optional[str] = None, choice_format: str = "letter"):
        self.template = custom_template if custom_template else self.TEMPLATES.get(template_name, self.TEMPLATES["basic"])
        self.choice_format = choice_format
    
    def _format_choices(self, choices: List[str]) -> str:
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"] if self.choice_format == "letter" else [str(i+1) for i in range(len(choices))]
        return "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    
    def format(self, context: str, question: str, choices: List[str], **kwargs) -> str:
        return self.template.format(context=context, question=question, choices_formatted=self._format_choices(choices), **kwargs)
    
    def format_with_answer(self, context: str, question: str, choices: List[str], answer_idx: int) -> str:
        prompt = self.format(context, question, choices)
        label = chr(ord('A') + answer_idx) if self.choice_format == "letter" else str(answer_idx + 1)
        return f"{prompt} {label}"


class PromptingPipeline:
    def __init__(
        self,
        model,
        tokenizer,
        template: Optional[PromptTemplate] = None,
        device: str = "cuda",
        max_prompt_tokens: Optional[int] = None,
        exemplars: Optional[List[Dict[str, Any]]] = None,
    ):
        self.model = model.to(device) if hasattr(model, 'to') else model
        self.tokenizer = tokenizer
        self.template = template or PromptTemplate("basic")
        self.device = device
        self.max_prompt_tokens = max_prompt_tokens
        self.exemplars = exemplars or []
        self._setup_choice_tokens()
    
    def _setup_choice_tokens(self):
        self.choice_tokens = {}
        for label in ["A", "B", "C", "D"]:
            for prefix in ["", " "]:
                token_ids = self.tokenizer.encode(prefix + label)
                if token_ids:
                    self.choice_tokens[label] = token_ids[-1]
                    break
    
    def _truncate_context_to_fit(
        self, context: str, question: str, choices: List[str], max_len: int
    ) -> str:
        """Truncate context so that context + question + choices fits within max_len tokens."""
        reserved = len(
            self.tokenizer.encode(self.template.format("", question, choices))
        )
        max_context_tokens = max(1, max_len - reserved)
        context_tokens = self.tokenizer.encode(context)
        if len(context_tokens) <= max_context_tokens:
            return context
        context_tokens = context_tokens[-max_context_tokens:]
        return self.tokenizer.decode(context_tokens)

    def _format_choice_input(self, context: str, question: str, choice: str) -> str:
        """Match fine-tuned format: same as MultipleChoiceQADataset._format_choice_input."""
        return f"{context}\n\nQuestion: {question}\n\nAnswer: {choice}"

    def _build_few_shot_prefix(self) -> str:
        if not self.exemplars:
            return ""
        blocks = []
        for ex in self.exemplars:
            blocks.append(self._format_choice_input(ex["context"], ex["question"], ex["answer_text"]))
        return "\n\n".join(blocks) + "\n\n"

    def _score_choice_logprob(self, context: str, question: str, choice: str, max_len: int) -> float:
        """Score choice by log P(choice | context + question + 'Answer: '). Matches fine-tune format."""
        few_shot = self._build_few_shot_prefix()
        prefix_fixed = f"\n\nQuestion: {question}\n\nAnswer: "
        choice_ids = self.tokenizer.encode(choice)
        fixed_ids = self.tokenizer.encode(few_shot + prefix_fixed)
        max_context_tokens = max(1, max_len - len(fixed_ids) - len(choice_ids))
        context_tokens = self.tokenizer.encode(context)
        if len(context_tokens) > max_context_tokens:
            context_tokens = context_tokens[-max_context_tokens:]
        context_trunc = self.tokenizer.decode(context_tokens)
        prefix = f"{few_shot}{context_trunc}{prefix_fixed}"
        full = prefix + choice
        prefix_ids = self.tokenizer.encode(prefix)
        full_ids = self.tokenizer.encode(full)
        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]
        if len(full_ids) <= len(prefix_ids):
            return float("-inf")
        input_ids = torch.tensor([full_ids[:-1]], device=self.device)
        logits = self.model(input_ids)[0]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        prefix_len = len(prefix_ids)
        total_logprob = 0.0
        for i, tid in enumerate(full_ids[1:]):
            if i + 1 >= prefix_len:
                total_logprob += log_probs[i, tid].item()
        return total_logprob

    @torch.no_grad()
    def predict_single(self, context: str, question: str, choices: List[str], return_probs: bool = False):
        self.model.eval()
        max_len = getattr(self.model, "context_length", 512)
        if self.max_prompt_tokens is not None:
            max_len = min(max_len, self.max_prompt_tokens)
        reserved = len(
            self.tokenizer.encode(f"\n\nQuestion: {question}\n\nAnswer: ")
        ) + 50
        max_context_tokens = max(1, max_len - reserved)
        context_tokens = self.tokenizer.encode(context)
        if len(context_tokens) > max_context_tokens:
            context = self.tokenizer.decode(context_tokens[-max_context_tokens:])
        scores = [
            self._score_choice_logprob(context, question, c, max_len)
            for c in choices
        ]
        probs = softmax(torch.tensor(scores), dim=-1)
        prediction = probs.argmax().item()
        if return_probs:
            return prediction, probs.tolist()
        return prediction
    
    @torch.no_grad()
    def predict_batch(self, examples: List[Dict[str, Any]], batch_size: int = 8) -> List[int]:
        return [self.predict_single(ex["context"], ex["question"], ex["choices"]) for ex in examples]


def evaluate_prompting(pipeline, examples: List[Dict[str, Any]], batch_size: int = 8) -> Dict[str, Any]:
    predictions = pipeline.predict_batch(examples, batch_size)
    correct = sum(1 for p, ex in zip(predictions, examples) if ex.get("answer", -1) >= 0 and p == ex["answer"])
    total = sum(1 for ex in examples if ex.get("answer", -1) >= 0)
    return {"accuracy": correct / total if total > 0 else 0.0, "predictions": predictions}
