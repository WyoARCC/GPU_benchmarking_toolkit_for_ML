# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training GPT2 with Wikitext103-v1 from HuggingFace
# Backend: Pytorch


import json

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from multiprocessing import Pool
from datasets import Dataset


class GPTData(torch.utils.data.Dataset):
    # noinspection PyPep8Naming
    def __init__(self, path: str, tokenizer_pp):
        self.data = json.load(open(path, 'r', encoding="utf-8"))
        self.tmpList = []
        self.tokenizer = tokenizer_pp
        for j in self.data["data"]:
            self.tmpList.append(j["text"])
        for idx, i in enumerate(self.tmpList):
            try:
                self.tmpList[idx] = "<bos>" + self.tmpList[idx + 1] + "<eos>"

            except:
                break

        # Convert to HuggingFace Dataset Type
        self.tmpList = Dataset.from_list(self.tmpList)
        print(self.tmpList)
        print("Tokenizing: " + path + "\n")
        self.tmpList_encoded = self.tokenizer.map(self.tmpList, batched=True)
        print("Finished Tokenizing..\n")

        self.input_ids = []
        self.attention_mask = []
        for batch in self.tmpList_encoded:
            self.input_ids.extend(batch["input_ids"])
            self.attention_mask.extend(batch["attention_mask"])

    def encode_text(self, text):
        return self.tokenizer(text, truncation=True, max_length=1024, padding="max_length", return_tensors="pt")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index]
