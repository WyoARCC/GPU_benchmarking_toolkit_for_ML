# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training BERT with glue from HuggingFace
# Backend: Pytorch
import concurrent.futures
import json
from torch.utils.data import Dataset
from tqdm import tqdm


class BERTDataLabled(Dataset):
    def __init__(self, json_path, tokenizer, context_length=128):
        self.tokenizer = tokenizer
        self.data = []
        self.labels = []
        self.max_context = context_length
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)["data"]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(self._tokenize, json_data), total=len(json_data)))
            self.data = [result["input_ids"] for result in results]
            self.labels = [result["label"] for result in results]

    def _tokenize(self, item):
        encoding = self.tokenizer(item["text"], truncation=True,
                                  max_length=self.max_context,
                                  padding="max_length",
                                  return_tensors='pt')
        encoding["label"] = item["label"]
        return encoding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]['input_ids'].squeeze()
        attention_mask = self.data[idx]['attention_mask'].squeeze()
        label = self.labels[idx]
        return input_ids, attention_mask, label
