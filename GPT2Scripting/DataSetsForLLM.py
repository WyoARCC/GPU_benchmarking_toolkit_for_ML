# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Validated on GPT2-Large With OpenWebText
# Backend: Pytorch
from datasets import load_dataset
import os

context_length = int(os.environ.get('MAX_TOK_LENGTH')) if os.environ.get('MAX_TOK_LENGTH') is not None else 128


class WikiTextDataset:
    def __init__(self, tokenizer, split):
        # Load the WikiText dataset with the specified split
        self.data = load_dataset('wikitext', 'wikitext-103-v1', split=split)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Tokenize the text at the given index using the tokenizer
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=context_length, padding=False,
                              return_tensors='pt')
        # Convert the tensors to long datatype and return them as a dictionary
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)


class BookCorpusDataset:
    def __init__(self, tokenizer, split='train'):
        # Load the BookCorpus dataset with the specified split
        self.data = load_dataset('bookcorpus', split=split)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Tokenize the text at the given index using the tokenizer
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=context_length, padding=False,
                              return_tensors='pt')
        # Convert the tensors to long datatype and return them as a dictionary
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)


class OpenWebTextDataset:
    def __init__(self, tokenizer, split='train'):
        # Load the OpenWebText dataset with the specified split
        self.data = load_dataset("openwebtext", split=split)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Tokenize the text at the given index using the tokenizer
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=context_length, padding=False,
                              return_tensors='pt')
        # Convert the tensors to long datatype and return them as a dictionary
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)


class PileDataset:
    def __init__(self, tokenizer, split='train'):
        # Load the 'all' subset of the Pile dataset with the specified split
        self.data = load_dataset("EleutherAI/pile", "all", split=split)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Tokenize the text at the given index using the tokenizer
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=context_length, padding=False,
                              return_tensors='pt')
        # Convert the tensors to long datatype and return them as a dictionary
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)


class RedPajamaDataset:
    def __init__(self, tokenizer, split='train'):
        # Load the RedPajama-Data-1T-Sample dataset with the specified split
        self.data = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split=split)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Tokenize the text at the given index using the tokenizer
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=context_length, padding=False,
                              return_tensors='pt')
        # Convert the tensors to long datatype and return them as a dictionary
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)


class OscarDataset:
    def __init__(self, tokenizer, split='train'):
        # Load the 'unshuffled_deduplicated_en' subset of the OSCAR dataset with the specified split
        self.data = load_dataset("oscar", "unshuffled_deduplicated_en", split=split)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Tokenize the text at the given index using the tokenizer
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=context_length, padding=False,
                              return_tensors='pt')
        # Convert the tensors to long datatype and return them as a dictionary
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)


class StarCoderDataset:
    def __init__(self, tokenizer, data_dir='python', split='train'):
        # Load the StarCoder dataset with the specified split and data directory
        self.data = load_dataset("bigcode/starcoderdata", data_dir=data_dir, split=split)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Tokenize the text at the given index using the tokenizer
        item = self.tokenizer(self.data[idx]['text'], truncation=True, max_length=context_length, padding=False,
                              return_tensors='pt')
        # Convert the tensors to long datatype and return them as a dictionary
        return {key: value.long() for key, value in item.items()}

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)