# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training GPT2 with Wikitext103-v1 from HuggingFace
# Backend: Pytorch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import os
import csv
from GPTData import GPTData
from GPTData_SingleThread import GPTData_SingleThread
from DataSetsForLLM import LoadWikiText, LoadOpenWebText  # , LoadOpenWebText
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
num_gpus = os.environ.get('NUM_GPUS', -1)
accelerator = Accelerator(mixed_precision='fp16')
criterion = CrossEntropyLoss()


def open_wikiText():
    if os.path.isfile('wikitext-103-v1-train.json'):
        print('Loading Training JSON File...')
    else:
        print('Generating Training JSON Files...')
        train_data = LoadWikiText('train')
        train_data.save_to_json('wikitext-103-v1-train.json')
        # Load Wikitext-103-v1 Validation Split and convert it to .json formatting
    if os.path.isfile('wikitext-103-v1-validation.json'):
        print('Loading Validation JSON File...')
    else:
        print('Generating Validation JSON Files...')
        validation_data = LoadWikiText('validation')
        validation_data.save_to_json('wikitext-103-v1-validation.json')
    # Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
    print("Preprocessing...")
    TrainingChatData = multi_core_GPTDATA('wikitext-103-v1-train.json', tokenizer)
    ValidatingChatData = multi_core_GPTDATA('wikitext-103-v1-validation.json', tokenizer)
    # Create distributed version of the dataset
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(TrainingChatData, batch_size=32, shuffle=True)
    ValidatingChatData = DataLoader(ValidatingChatData, batch_size=32, pin_memory=False)

    return TrainingChatData, ValidatingChatData


def open_OpenWebText():
    if os.path.isfile('openwebtext-train.json'):
        print('Loading Training JSON File...')
    else:
        print('Generating Training JSON Files...')
        train_data = LoadOpenWebText()
        train_data.save_to_json('openwebtext-train.json')

    if os.path.isfile('wikitext-103-v1-validation.json'):
        print('Loading Validation JSON File...')
    else:
        print('Generating Validation JSON Files...')
        validation_data = LoadWikiText('validation')
        validation_data.save_to_json('wikitext-103-v1-validation.json')

    print("Preprocessing...")
    TrainingChatData = multi_core_GPTDATA('openwebtext-train.json', tokenizer)
    ValidatingChatData = multi_core_GPTDATA('openwebtext-validation.json', tokenizer)

    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(TrainingChatData, batch_size=32, shuffle=True)
    ValidatingChatData = DataLoader(ValidatingChatData, batch_size=32, pin_memory=False)

    return TrainingChatData, ValidatingChatData


def multi_core_GPTDATA(filename: str, tokenizer_mc):
    # print("error print if >1")
    temp_set = GPTData(json_path=filename, tokenizer=tokenizer_mc)
    return temp_set


# yes
def single_core_GPTDATA(filename: str, tokenizer_sc):
    tmp_set = GPTData_SingleThread(path=filename, tokenizer_pp=tokenizer_sc)
    return tmp_set


def GPT2_Tokenizer():
    tokenizerGrab = GPT2TokenizerFast.from_pretrained("gpt2-medium")
    tokenizerGrab.pad_token = '<pad>'
    tokenizerGrab.eos_token = '<eos>'
    tokenizerGrab.bos_token = '<bos>'
    return tokenizerGrab


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
    val_loss = 0
    num_iterations = 0
    accumulation_steps = 3
    torch.set_default_tensor_type("torch.FloatTensor")
    for epochR in range(epoch_pass):
        # Training loop
        start = time.time()
        for batch in tqdm(trainer_data, desc=f'Training Epoch {epochR}, Batch', leave=True):
            if batch is None:
                # Skip the last empty batch (As the multicore Encoder returns NoneType for last index)
                continue
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model_pass(inputs)
            loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), targets.view(-1))
            accelerator.backward(loss)
            num_iterations += 1
            # Gradient Accumulation
            if num_iterations == accumulation_steps:
                optimizer.step()
                scheduler_pass.step(loss)
                num_iterations = 0

        accelerator.free_memory()
        end = time.time()
        epochTime = end - start

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
                writer.writerow(['Epoch', 'Validation Loss', 'Time'])
            writer.writerow([epochR + 1, val_loss, epochTime])

        accelerator.wait_for_everyone()


if __name__ == '__main__':
    # tokenizer Declaration and special token Declaration
    tokenizer = GPT2_Tokenizer()
    # Model Declaration
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    model.resize_token_embeddings(len(tokenizer))
    # Load Data
    TrainChatData, ValidationChatData = open_wikiText()
    #TrainChatData, ValidationChatData = open_OpenWebText()
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
