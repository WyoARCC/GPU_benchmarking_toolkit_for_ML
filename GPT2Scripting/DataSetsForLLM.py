# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training GPT2 with Wikitext103-v1 from HuggingFace
# Backend: Pytorch
from datasets import load_dataset
import json
import os


# Loads Wikitext-103-v1 from Hugging face DataSets online repository and converts it to json formatting
# Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. Pointer Sentinel Mixture Models
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


# Loads OpenAi's OpenWeb-text from Hugging face DataSets online repository and converts it to json formatting
# @misc{Gokaslan2019OpenWeb,
# title={OpenWebText Corpus},
# author={Aaron Gokaslan and Vanya Cohen},
# howpublished{\url{http://Skylion007.github.io/OpenWebTextCorpus}},
# year={2019}
# }
class LoadOpenWebText:
    def __init__(self, split):
        self.split = split
        self.data = load_dataset('openwebtext', split=self.split)
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
