# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training GPT2 with Wikitext103-v1 from HuggingFace
# Backend: Pytorch
from datasets import load_dataset


class WikiTextDataset:
    def __init__(self, tokenizer, split):
        self.data = load_dataset('wikitext', 'wikitext-103-v1', split=split)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=128, padding=False,
                              return_tensors='pt')
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        return len(self.data)


class BookCorpusDataset:
    def __init__(self, tokenizer, split='train'):
        self.data = load_dataset('bookcorpus', split=split)
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
