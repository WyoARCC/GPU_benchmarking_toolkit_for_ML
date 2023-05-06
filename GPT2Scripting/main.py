# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training GPT2 with Wikitext103-v1 from HuggingFace
# Backend: Pytorch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import gc
import csv
from GPTData import GPTData
from DataSetsForLLM import LoadWikiText
from tqdm import tqdm
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.cuda
from accelerate import Accelerator
import torch

# Define Device for Training
num_gpus = os.environ.get('NUM_GPUS', -1)
accelerator = Accelerator(mixed_precision='fp16', num_gpus=int(num_gpus)
criterion = CrossEntropyLoss()


def train_and_validate(trainer_data, val_data, model_pass, optimizer, scheduler_pass, epoch_pass):
    model_pass.train()
    train_loss = 0
    val_loss = 0
    num_iterations = 0
    accumulation_steps = 16
    torch.set_default_tensor_type("torch.FloatTensor")
    # Training loop
    for batch in tqdm(trainer_data, desc='Training Batch', leave=False):
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model_pass(inputs)
        loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), targets.view(-1))
        accelerator.backward(loss)
        num_iterations += 1
        if num_iterations == accumulation_steps:
            optimizer.step()
            scheduler_pass.step(loss, epoch_pass)
            num_iterations = 0
        del loss, outputs, inputs, targets
        gc.collect()
    if num_iterations != 0:
        optimizer.step()
        scheduler_pass.step(loss, epoch_pass)
    torch.cuda.empty_cache()
    accelerator.free_memory()

    # Validation loop
    model_pass.eval()
    for batch in tqdm(val_data, desc='Validation Batch', leave=False):
        inputs, targets = batch
        outputs = model_pass(inputs)
        loss = criterion(outputs, targets)
        accelerator.backward(loss)
        val_loss += loss.item()

    val_loss /= len(val_data)

    # Save model and write validation loss to CSV file
    accelerator.save(model_pass.state_dict(), f"model_state_{epoch_pass}.pt")
    with open('validation_results.csv', mode='a', newline='') as results_file:
        writer = csv.writer(results_file)
        if epoch_pass == 0:
            writer.writerow(['Epoch', 'Validation Loss'])
        writer.writerow([epoch_pass + 1, val_loss])


# Have the model conduct Inferences
def infer(inp):
    input_formatted = "<bos>" + inp + "<eos>"
    test = tokenizer(input_formatted)
    output = model.generate(**test)
    return tokenizer.decode(output[0])

    # if torch.cuda.is_available() else "cpu")


# tokenizer Declaration and special token Declaration
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
tokenizer.pad_token = '<pad>'
tokenizer.eos_token = '<eos>'
tokenizer.bos_token = '<bos>'

# Model Declaration
model = GPT2LMHeadModel.from_pretrained("gpt2-large")
model.resize_token_embeddings(len(tokenizer))
model.to(accelerator.device)
# Load Wikitext-103-v1 Train Split and convert it to .json formatting
if os.path.exists('wikitext-103-v1-train.json'):
    print('Loading Training JSON File...')
    pass
else:
    print('Generating Training JSON File...')
    train_data = LoadWikiText('train')
    train_data.save_to_json('wikitext-103-v1-train.json')

# Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
TrainChatData = GPTData('wikitext-103-v1-train.json', tokenizer, 20)

# Create distributed version of the dataset
print("Distributing Data Set...")
TrainChatData = DataLoader(TrainChatData, batch_size=32, pin_memory=True, shuffle=False)

# Load Wikitext-103-v1 Validation Split and convert it to .json formatting
if os.path.exists('wikitext-103-v1-validation.json'):
    print('Loading Validation JSON File...')
    pass
else:
    validation_data = LoadWikiText('validation')
    validation_data.save_to_json('wikitext-103-v1-validation.json')
# Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
ValidationChatData = GPTData('wikitext-103-v1-validation.json', tokenizer, 20)

# Create distributed version of the dataset
ValidationChatData = DataLoader(ValidationChatData, batch_size=32, pin_memory=False)

# Set Up Training
optim = Adam(model.parameters(), lr=5e-6)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2, eta_min=0.0001)
# Distributed passthrough
model, optim, scheduler, TrainChatData, ValidationChatData = accelerator.prepare(model,
                                                                                 optim,
                                                                                 scheduler,
                                                                                 TrainChatData,
                                                                                 ValidationChatData)
# Call Training Function, Will write a CSV file
# Train loop
for epoch in range(5):
    train_and_validate(TrainChatData, ValidationChatData, model, optim, scheduler, epoch)
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), f"model_state_{epoch}.pt")
