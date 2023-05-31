# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training GPT2 with Wikitext103-v1 from HuggingFace
# Backend: Pytorch
import json
import os
from datasets import load_dataset


class LoadDataset:
    def __init__(self, dataset_name, split):
        self.data = load_dataset(dataset_name, split=split)
        self.examples = [{'text': example['sentence']} for example in self.data]

    @staticmethod
    def create_directory(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    def save_to_json(self, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'data': self.examples}, f, ensure_ascii=False, indent=4)


class LoadWikiText:
    def __init__(self, split):
        self.split = split
        self.data = load_dataset('wikitext', 'wikitext-103-v1', split=self.split)
        self.examples = [{'text': example['text']} for example in self.data]

    @staticmethod
    def create_directory(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    def save_to_json(self, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'data': self.examples}, f, ensure_ascii=False, indent=4)


class LoadOpenWebText:
    def __init__(self):
        self.data = load_dataset('openwebtext')
        self.examples = [{'text': example} for example in self.data['text']]

    @staticmethod
    def create_directory(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    def save_to_json(self, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'data': self.examples}, f, ensure_ascii=False, indent=4)


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
