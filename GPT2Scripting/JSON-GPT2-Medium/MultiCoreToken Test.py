# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training GPT2 with Wikitext103-v1 from HuggingFace
# Backend: Pytorch

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from torch.utils.data import DataLoader
from GPTData import GPTData

# Load the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = '<pad>'
tokenizer.eos_token = '<eos>'
tokenizer.bos_token = '<bos>'

    # Model Declaration
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Create an instance of the GPTData dataset
dataset = GPTData(path='wikitext-103-v1-train.json', tokenizer_pp=tokenizer, batch_size=256, num_workers=19)
dataset.load_data()
# Create a DataLoader for the dataset
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(dataloader)
# Iterate over the batches in the DataLoader
for batch in dataloader:
    input_ids, attention_mask = batch
    print("Input IDs:", input_ids.shape)  # Print the shape of the input IDs tensor
    print("Attention Mask:", attention_mask.shape)  # Print the shape of the attention mask tensor
    break  # Stop after the first batch for this example
