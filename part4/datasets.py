"""
Dataset classes for pre-training and fine-tuning.
Example submission.
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class PretrainingDataset(Dataset):
    def __init__(self, file_path: str | Path, tokenizer, max_length: int = 256, stride: int | None = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.token_ids = tokenizer.encode(text)
        if len(self.token_ids) <= max_length:
            self.num_sequences = 1
        else:
            self.num_sequences = (len(self.token_ids) - max_length) // self.stride + 1
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = idx * self.stride
        end_idx = min(start_idx + self.max_length + 1, len(self.token_ids))
        sequence = self.token_ids[start_idx:end_idx]
        if len(sequence) < self.max_length + 1:
            sequence = sequence + [0] * (self.max_length + 1 - len(sequence))
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = torch.tensor(sequence[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


class MultipleChoiceQADataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 256, num_choices: int = 4):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_choices = num_choices
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _format_choice_input(self, context: str, question: str, choice: str) -> str:
        return f"{context}\n\nQuestion: {question}\n\nAnswer: {choice}"
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        context = example["context"]
        question = example["question"]
        choices = example["choices"]
        answer = example.get("answer", -1)
        
        all_input_ids = []
        all_attention_masks = []
        
        for choice in choices:
            text = self._format_choice_input(context, question, choice)
            token_ids = self.tokenizer.encode(text)
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            attention_mask = [1] * len(token_ids)
            padding_length = self.max_length - len(token_ids)
            token_ids = token_ids + [0] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            all_input_ids.append(token_ids)
            all_attention_masks.append(attention_mask)
        
        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_attention_masks, dtype=torch.long),
            "labels": torch.tensor(answer, dtype=torch.long),
        }
    
    @classmethod
    def from_json(cls, file_path: str | Path, tokenizer, **kwargs) -> "MultipleChoiceQADataset":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data, tokenizer, **kwargs)


def create_pretraining_dataloader(file_path, tokenizer, batch_size=8, max_length=256, stride=None, shuffle=True, num_workers=0, sampler=None):
    dataset = PretrainingDataset(file_path, tokenizer, max_length, stride)
    if sampler is not None:
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, pin_memory=True)


def create_qa_dataloader(data, tokenizer, batch_size=4, max_length=256, num_choices=4, shuffle=True, num_workers=0, sampler=None):
    if isinstance(data, (str, Path)):
        dataset = MultipleChoiceQADataset.from_json(data, tokenizer, max_length=max_length, num_choices=num_choices)
    else:
        dataset = MultipleChoiceQADataset(data, tokenizer, max_length=max_length, num_choices=num_choices)
    if sampler is not None:
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, pin_memory=True)
