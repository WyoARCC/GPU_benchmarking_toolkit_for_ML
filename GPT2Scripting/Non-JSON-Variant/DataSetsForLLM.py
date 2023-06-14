# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training GPT2 with Wikitext103-v1 from HuggingFace
# Backend: Pytorch
from datasets import load_dataset
import json
import os


class WikiTextDataset:
    def __init__(self, split, tokenizer):
        self.data = load_dataset('wikitext', 'wikitext-103-v1', split=split)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=128, padding=False,
                              return_tensors='pt')
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        return len(self.data)


class OpenWebTextDataset:
    def __init__(self, tokenizer, split='train'):
        self.data = load_dataset("openwebtext", split=split)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=128, padding=False,
                              return_tensors='pt')
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        return len(self.data)


class LoadOscar:
    def __init__(self, split):
        self.split = split
        self.data = load_dataset('oscar', 'unshuffled_deduplicated_en', split=self.split)
        self.examples = [{'text': example} for example in self.data['text']]

    @staticmethod
    def create_directory(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    def save_to_json(self, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'data': self.examples}, f, ensure_ascii=False, indent=4)
