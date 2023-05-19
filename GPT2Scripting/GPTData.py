# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training GPT2 with Wikitext103-v1 from HuggingFace
# Backend: Pytorch
# import itertools
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os


class GPTData(Dataset):
    def __init__(self, path: str, tokenizer_pp, batch_size=64, num_workers=10):
        self.path = path
        self.tokenizer_pp = tokenizer_pp
        self.batch_size = batch_size
        self.num_workers = num_workers
        # sets what portion of the dataset to sample (-1 is whole dataset)
        self.data_set_size = 10000
        self.data = None
        self.chunk_index = 0
        self.chunk_size = batch_size * num_workers
        self.tmpList = None

    def load_data(self):
        with open(self.path, 'r', encoding="utf-8") as file:
            self.data = json.load(file)["data"]
            self.tmpList = [j["text"] for j in self.data]
            self.tmpList = self.tmpList[:self.data_set_size]  # change back to :-1

    def get_chunk(self):
        if self.data is None:
            self.load_data()

        start_index = self.chunk_index * self.chunk_size
        end_index = (self.chunk_index + 1) * self.chunk_size

        if start_index < len(self.tmpList):
            chunk = self.tmpList[start_index:end_index]
            self.chunk_index += 1
            return chunk
        else:
            return None

    def tokenize_batch(self, batch):
        return self.tokenizer_pp.batch_encode_plus(batch, truncation=True, max_length=1024,
                                                   padding="max_length", return_tensors='pt')

    def __len__(self):
        if self.tmpList is None:
            return 0  # or return an appropriate default length
        return len(self.tmpList)

    def __iter__(self):
        total_texts = len(self.tmpList) - 1
        pbar = tqdm(total=total_texts, desc='Tokenizing dataset')

        def tokenization_task(batch_arg):
            results = []
            for texts in batch_arg:
                results.append(self.tokenize_batch(["<bos>" + texts + "<eos>"]))
            return results

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for idx, text in enumerate(self.tmpList):
                batch = [text, self.tmpList[idx + 1]]
                futures.append(executor.submit(tokenization_task, batch))
                pbar.update(1)
            finished_texts = 0
            for future in futures:
                yield from future.result()
                finished_texts += 1

        pbar.close()

    def save_encoded_chunk(self, chunk):
        encoded_chunk = self.tokenize_batch(chunk)
        save_path = os.path.join('__numpycache__', str(self.chunk_index - 1), 'encoded_chunk.npy')
        np.save(save_path, encoded_chunk)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of range")

        batch_idx = index // self.batch_size
        data_idx = index % self.batch_size

        try:
            if data_idx == 0:
                chunk = self.get_chunk()
                if chunk is not None:
                    self.save_encoded_chunk(chunk)

            encoded_chunk = np.load(os.path.join('__numpycache__', str(batch_idx) + '.npy'), allow_pickle=True)
            chunk_data = encoded_chunk[data_idx]
            input_ids = chunk_data["input_ids"][index]
            attention_mask = chunk_data["attention_mask"][index]

            return input_ids, attention_mask

        except FileNotFoundError:
            return torch.zeros(1024, dtype=torch.long), torch.zeros(1024, dtype=torch.long)
        except IndexError:
            return torch.zeros(1024, dtype=torch.long), torch.zeros(1024, dtype=torch.long)

    # 1 (DONE) Write a lazy reader for the JSON File to load batches of the dataset and process them at a time rather
    # than loading the entire dataset into memory

    # 2 (DONE)
    # Encoding
    # Encoder should take this generator and yield back encoded parts

    # 3 (TESTING)
    # Saving encoded files As the files will be too large to fit in RAM memory, you should save them to disk (or use
    # somehow as they are generated).
