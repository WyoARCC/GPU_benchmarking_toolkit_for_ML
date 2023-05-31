# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training GPT2 with Wikitext103-v1 from HuggingFace
# Backend: Pytorch
# import itertools


# Name: Tyson Limato    Date: 5/29/2023
# Choice of ProcessPoolExecutor vs ThreadPoolExecutor. Previously I was using a ThreadPoolExecutor to submit tasks to
# a thread pool. This can be a good way to parallelize work, especially for I/O-bound tasks. However, Python's Global
# Interpreter Lock (GIL) can prevent true parallel execution of CPU-bound tasks. Given the tokenization process is
# generally CPU-bound and performance is my primary concern, I decided to use a ProcessPoolExecutor instead,
# which uses separate processes to bypass the GIL. Using processes is considered to have higher overhead than threads
# due to inter-process communication, with the benefit of true task parallelization which is advantageous in a high
# bandwidth HPC environment

import concurrent.futures
import json
from torch.utils.data import Dataset
from tqdm import tqdm


class BERTData(Dataset):
    def __init__(self, json_path, tokenizer, context_length=128):
        self.tokenizer = tokenizer
        self.data = []
        self.max_context = context_length
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)["data"]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            self.data = list(tqdm(executor.map(self._tokenize, json_data), total=len(json_data)))

    def _tokenize(self, item):
        return self.tokenizer(item["text"], truncation=True,
                              max_length=self.max_context,
                              padding="max_length",
                              return_tensors='pt')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # noinspection PyTypeChecker
        return self.data[idx]['input_ids'].squeeze(), self.data[idx]['attention_mask'].squeeze()
