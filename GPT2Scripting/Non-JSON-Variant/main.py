# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training GPT2 with Wikitext103-v1 from HuggingFace
# Backend: Pytorch
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, AutoTokenizer

import os
import csv

from DataSetsForLLM import WikiTextDataset, OpenWebTextDataset
from tqdm import tqdm
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.cuda
from accelerate import Accelerator
# import torch
import time

# import torch.multiprocessing as mp

# Define Device for Training
num_gpus = os.environ.get('CUDA_VISIBLE_DEVICES') if not None else 1
accelerator = Accelerator(mixed_precision='fp16')
criterion = CrossEntropyLoss()


def open_wikiText():
    print('Loading Training Data Files...')
    train_data = WikiTextDataset('train', tokenizer)

    # Load Wikitext-103-v1 Validation Split and convert it to .json formatting
    print('Loading Validation Data Files...')
    validation_data = WikiTextDataset('validation', tokenizer)
    # Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
    print("Preprocessing...")
    # Create distributed version of the dataset
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collate_fn)
    ValidatingChatData = DataLoader(validation_data, batch_size=8, collate_fn=collate_fn)
    return TrainingChatData, ValidatingChatData


def open_WebText():
    print('Loading Training Data Files...')
    train_data = OpenWebTextDataset(tokenizer)
    # Load Wikitext-103-v1 Validation Split and convert it to .json formatting
    print('Loading Validation Data Files...')
    validation_data = WikiTextDataset('validation', tokenizer)
    # Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
    print("Preprocessing...")
    # Create distributed version of the dataset
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collate_fn)
    ValidatingChatData = DataLoader(validation_data, batch_size=8, collate_fn=collate_fn)
    return TrainingChatData, ValidatingChatData


def GPT2_Tokenizer():
    tokenizerGrab = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizerGrab.pad_token = '<pad>'
    tokenizerGrab.eos_token = '<eos>'
    tokenizerGrab.bos_token = '<bos>'
    return tokenizerGrab


def collate_fn(batch):
    input_ids = [item['input_ids'].squeeze() for item in batch]
    attention_masks = [item['attention_mask'].squeeze() for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)

    return input_ids, attention_masks


# Have the model conduct Inferences

def infer(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
        attention_mask = attention_mask.to("cuda")
        model.to("cuda")
    outputs = model.generate(inputs, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_length=150)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def train_and_validate(trainer_data, val_data, model_pass, optimizer, scheduler_pass, epoch_pass):
    model_pass.train()
    max_grad_norm = 1
    val_loss = 0
    num_iterations = 0
    accumulation_steps = 1
    torch.set_default_tensor_type("torch.FloatTensor")
    for epochR in range(epoch_pass):
        # Training loop
        start = time.time()
        batch_size = 0
        for batch in tqdm(trainer_data, desc=f'Training Epoch {epochR}, Batch', leave=True):
            if batch is None:
                # Skip the last empty batch (As the multicore Encoder returns NoneType for last index)
                continue
            inputs, targets = batch
            batch_size = len(inputs)
            optimizer.zero_grad()
            outputs = model_pass(inputs)
            loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), targets.view(-1))
            accelerator.backward(loss)
            num_iterations += 1
            # Gradient Accumulation
            if num_iterations == accumulation_steps:
                # Gradient clipping
                clip_grad_norm_(model_pass.parameters(), max_grad_norm)
                optimizer.step()
                scheduler_pass.step(loss)
                num_iterations = 0

        accelerator.free_memory()
        end = time.time()
        epochTime = end - start

        throughput = batch_size * len(trainer_data) / epochTime

        # Validation loop
        model_pass.eval()
        for batch in tqdm(val_data, desc='Validation Batch', leave=True):
            if batch is None:
                continue  # Skip the last empty batch
            inputs, targets = batch
            outputs = model_pass(inputs)
            loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), targets.view(-1))
            val_loss += accelerator.gather(loss)

        torch.cuda.empty_cache()
        accelerator.free_memory()

        val_loss /= len(val_data)
        print(f"Test Response for Epoch: {epochR}")
        TempModelGeneration = infer(
            "Albert Einstein was ")
        print(TempModelGeneration)

        # Save model and write validation loss to CSV file
        accelerator.save(model_pass.state_dict(), f"model_state_{epochR}.pt")
        with open('validation_results.csv', mode='a', newline='') as results_file:
            writer = csv.writer(results_file)
            if epochR == 0:
                writer.writerow(['Epoch', 'Validation Loss', 'Time', 'Throughput'])
            writer.writerow([epochR + 1, val_loss, epochTime, throughput])

        accelerator.wait_for_everyone()


if __name__ == '__main__':
    # tokenizer Declaration and special token Declaration
    tokenizer = GPT2_Tokenizer()
    # Model Declaration
    model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    model.resize_token_embeddings(len(tokenizer))
    # Load Data
    TrainChatData, ValidationChatData = open_WebText()
    # TrainChatData, ValidationChatData = open_OpenWebText()
    # Define Optimizer and Scheduler
    optim = Adam(model.parameters(), lr=5e-6)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2, eta_min=0.0001)
    # Accelerate Distributed passthrough
    model, optim, scheduler, TrainChatData, ValidationChatData = accelerator.prepare(model,
                                                                                     optim,
                                                                                     scheduler,
                                                                                     TrainChatData,
                                                                                     ValidationChatData)
    try:
        # Call Training Function (Will write a CSV file)
        epoch = 3
        # Set Token length per Text Entry
        # (Entries Longer than specified number will be truncated and Entries Shorter will be Padded)
        # GPT2 has a max length of 1024 tokens
        # According to OpenAI, the conversion rate of character to token is 4:1
        # Cite: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        os.environ['max_tok_length'] = str(256)
        # Training RunTime
        print("Fine-tuning...")
        train_and_validate(TrainChatData, ValidationChatData, model, optim, scheduler, epoch)
        print("successful fine-tuning...")
        print("Testing Model Training Results With Validation Prompt...")
        for x in range(10):
            ModelGeneration = infer(
                "Albert Einstein was ")
            print(ModelGeneration)

    except KeyboardInterrupt:
        print("Aborted by the User")
