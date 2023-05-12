# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training GPT2 with Wikitext103-v1 from HuggingFace
# Backend: Pytorch

import json

from torch.utils.data import Dataset
from tqdm import tqdm


class GPTData(Dataset):
    # noinspection PyPep8Naming
    def __init__(self, path: str, tokenizer_pp):
        self.data = json.load(open(path, 'r', encoding="utf-8"))
        self.tmpList = []
        for j in self.data["data"]:
            self.tmpList.append(j["text"])
        for idx, i in enumerate(self.tmpList):
            try:
                self.tmpList[idx] = "<bos>" + self.tmpList[idx + 1] + "<eos>"

            except:
                break
        # Don't access the "text" label
        self.tmpList = self.tmpList[:500000]  # change back to :-1

        self.tmpList_encoded = []
        for batch_idx in tqdm(range(0, len(self.tmpList), 64), desc='Tokenizing...'):
            batch = self.tmpList[batch_idx:batch_idx + 64]
            tmpList_encoded_batch = tokenizer_pp(batch, truncation=True,
                                                 max_length=1024,
                                                 padding="max_length",
                                                 return_tensors='pt')
            self.tmpList_encoded.append(tmpList_encoded_batch)

        self.input_ids = []
        self.attention_mask = []
        for batch in self.tmpList_encoded:
            self.input_ids.extend(batch["input_ids"])
            self.attention_mask.extend(batch["attention_mask"])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index]
