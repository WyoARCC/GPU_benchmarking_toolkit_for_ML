# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training GPT2 with Wikitext103-v1 from HuggingFace
# Backend: Pytorch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from GPTData import GPTData
from DataSetsForLLM import LoadWikiText
import torch
from tqdm import tqdm
import os
import csv
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.data import DataLoader
torch.cuda.empty_cache()

# Initialize the process group
dist.init_process_group(
    backend="nccl",
    init_method="env://",  # Alternative: "tcp://%h:%x"
    world_size=int(os.environ['WORLD_SIZE']),
    rank=int(os.environ['RANK'])
)


# noinspection PyShadowingNames
def train_and_validate(train_data, val_data, model_pass, optimizer):
    epochs = 10
    model_pass = DistributedDataParallel(model_pass)  # integrate DataParallel
    for epoch in tqdm(range(epochs), desc='Epoch'):
        # Training loop
        for step, batch in tqdm(train_data, desc='Training Batch', leave=False):
            step = step.to(device)
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model_pass(step, attention_mask=batch, labels=step).loss
            loss.backward()
            optimizer.step()

        # Validation loop
        model_pass.eval()
        val_loss = 0
        with torch.no_grad():
            for step, batch in tqdm(val_data, desc='Validation Batch', leave=False):
                step = step.to(device)
                batch = batch.to(device)
                loss = model_pass(step, attention_mask=batch, labels=step).loss
                val_loss += loss.item()
            val_loss /= len(val_data)

        # Save model and write validation loss to CSV file
        torch.save(model_pass.state_dict(), f"model_state_{epoch}.pt")
        with open('validation_results.csv', mode='a', newline='') as results_file:
            writer = csv.writer(results_file)
            if epoch == 0:
                writer.writerow(['Epoch', 'Validation Loss'])
            writer.writerow([epoch + 1, val_loss])


# Have the model conduct Inferences
def infer(inp):
    input_formatted = "<bos>" + inp + "<eos>"
    test = tokenizer(input_formatted)
    output = model.generate(**test)
    return tokenizer.decode(output[0])


# Define Device for Training
device = torch.device("cuda")  # if torch.cuda.is_available() else "cpu")
rank = dist.get_rank()
torch.cuda.set_device(rank)
# tokenizer Declaration and special token Declaration
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
tokenizer.pad_token = '<pad>'
tokenizer.eos_token = '<eos>'
tokenizer.bos_token = '<bos>'

# Model Declaration
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
model.resize_token_embeddings(len(tokenizer))

# Wrap the model with DataParallel
model = DistributedDataParallel(model, device_ids=[rank])

# Move the model to the device
model = model.to(device)

# Load Wikitext-103-v1 Train Split and convert it to .json formatting
if os.path.exists('wikitext-103-v1-train.json'):
    print('Loading Training JSON File...')
    pass
else:
    print('Generating Training JSON File...')
    train_data = LoadWikiText('train')
    train_data.save_to_json('wikitext-103-v1-train.json')

# Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
TrainChatData = GPTData('wikitext-103-v1-train.json', tokenizer, 10)

# Create distributed version of the dataset
print("Distributing Data Set...")
train_sampler = torch.utils.data.distributed.DistributedSampler(
    TrainChatData,
    num_replicas=int(os.environ['WORLD_SIZE']),
    rank=rank,
    shuffle=True
)
TrainChatData = DataLoader(TrainChatData, batch_size=8, sampler=train_sampler)

# Load Wikitext-103-v1 Validation Split and convert it to .json formatting
if os.path.exists('wikitext-103-v1-validation.json'):
    print('Loading Validation JSON File...')
    pass
else:
    validation_data = LoadWikiText('validation')
    validation_data.save_to_json('wikitext-103-v1-validation.json')
# Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
ValidationChatData = GPTData('wikitext-103-v1-validation.json', tokenizer, 10)

# Create distributed version of the dataset
val_sampler = torch.utils.data.distributed.DistributedSampler(
    ValidationChatData,
    num_replicas=int(os.environ['WORLD_SIZE']),
    rank=rank,
    shuffle=False
)
ValidationChatData = DataLoader(ValidationChatData, batch_size=8, sampler=val_sampler)

# Set Up Training
model.train()
optim = Adam(model.parameters(), lr=5e-6)

# Call Training Function, Will write a CSV file
train_and_validate(TrainChatData, ValidationChatData, model, optim)
