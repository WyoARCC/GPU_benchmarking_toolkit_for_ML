# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training GPT2 with Wikitext103-v1 from HuggingFace
# Backend: Pytorch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import os
import csv
from GPTData import GPTData
from GPTData_SingleThread import GPTData_SingleThread
from DataSetsForLLM import LoadWikiText
from tqdm import tqdm
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.cuda
from accelerate import Accelerator
import torch
from multiprocessing import freeze_support
import time

# Define Device for Training
num_gpus = os.environ.get('NUM_GPUS', -1)
accelerator = Accelerator(mixed_precision='fp16')
criterion = CrossEntropyLoss()


def multi_core_GPTDATA(filename: str, tokenizer_mc):
    # print("error print if >1")
    temp_set = GPTData(path=filename, tokenizer_pp=tokenizer_mc, batch_size=32, num_workers=5)
    temp_set.load_data()
    return temp_set


def single_core_GPTDATA(filename: str, tokenizer_sc):
    tmp_set = GPTData_SingleThread(path=filename, tokenizer_pp=tokenizer_sc)
    return tmp_set


# Have the model conduct Inferences
def infer(prompt):
    input_formatted = "<bos>" + prompt + "<eos>"
    inputs = tokenizer.encode(input_formatted, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs, max_length=100)
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
            if num_iterations == accumulation_steps:
                optimizer.step()
                scheduler_pass.step(loss)
                num_iterations = 0
            # del loss, inputs, targets, outputs
            # torch.cuda.empty_cache()
            # accelerator.free_memory()
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
            accelerator.backward(loss)
            val_loss += loss.item()

        torch.cuda.empty_cache()
        accelerator.free_memory()

        val_loss /= len(val_data)

        # Save model and write validation loss to CSV file
        accelerator.save(model_pass.state_dict(), f"model_state_{epochR}.pt")
        with open('validation_results.csv', mode='a', newline='') as results_file:
            writer = csv.writer(results_file)
            if epochR == 0:
                writer.writerow(['Epoch', 'Validation Loss', 'Time'])
            writer.writerow([epochR + 1, val_loss, epochTime])

        accelerator.wait_for_everyone()
        # unwrapped_model = accelerator.unwrap_model(model)
        # torch.save(unwrapped_model.state_dict(), f"model_state_{epochR}.pt")


if __name__ == '__main__':
    freeze_support()
    # tokenizer Declaration and special token Declaration
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = '<pad>'
    tokenizer.eos_token = '<eos>'
    tokenizer.bos_token = '<bos>'

    # Model Declaration
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model.to(accelerator.device)
    # Load Wikitext-103-v1 Train Split and convert it to .json formatting
    if os.path.isfile('wikitext-103-v1-train.json'):
        print('Loading Training JSON File...')
    else:
        print('Generating Training JSON File...')
        train_data = LoadWikiText('train')
        train_data.save_to_json('wikitext-103-v1-train.json')

    # Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
    TrainChatData = multi_core_GPTDATA('wikitext-103-v1-train.json', tokenizer)

    # Create distributed version of the dataset
    print("Distributing Data Set...")
    TrainChatData = DataLoader(TrainChatData, batch_size=6, shuffle=True)

    # Load Wikitext-103-v1 Validation Split and convert it to .json formatting
    if os.path.isfile('wikitext-103-v1-validation.json'):
        print('Loading Validation JSON File...')
    else:
        validation_data = LoadWikiText('validation')
        validation_data.save_to_json('wikitext-103-v1-validation.json')
    # Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
    ValidationChatData = multi_core_GPTDATA('wikitext-103-v1-validation.json', tokenizer)

    # Create distributed version of the dataset
    ValidationChatData = DataLoader(ValidationChatData, batch_size=6, pin_memory=False)

    # Set Up Training
    optim = Adam(model.parameters(), lr=5e-6)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2, eta_min=0.0001)
    # Distributed passthrough
    # noinspection PyTypeChecker
    model, optim, scheduler, TrainChatData, ValidationChatData = accelerator.prepare(model,
                                                                                     optim,
                                                                                     scheduler,
                                                                                     TrainChatData,
                                                                                     ValidationChatData)
    # Call Training Function, Will write a CSV file
    # Train loop
    epoch = 3
    print("Fine-tuning...")
    train_and_validate(TrainChatData, ValidationChatData, model, optim, scheduler, epoch)
    print("successful fine-tuning...")
    print("Testing Model Training Results With Validation Prompt...")
    ModelGeneration = infer(
        "Generate a coherent and grammatically correct paragraph of text on the topic of the United States of "
        "America. Make sure the generated text demonstrates a deep understanding of the subject matter and maintains "
        "a consistent tone throughout.")
    print(ModelGeneration)
