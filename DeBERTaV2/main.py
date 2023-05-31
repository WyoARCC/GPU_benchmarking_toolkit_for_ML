# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training BERT with glue from HuggingFace
# Backend: Pytorch
from transformers import DebertaTokenizerFast, DebertaForMaskedLM
import os
import csv
from BERTData import BERTData
from LabeledBERTData import BERTDataLabled
from BERTData_SingleThread import BERTData_SingleThread
from DataSetsForLLM import LoadWikiText, LoadOpenWebText, LoadDataset
from tqdm import tqdm
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.cuda
from accelerate import Accelerator
# import torch
import time
import torch.multiprocessing as mp

# Define Device for Training
num_gpus = os.environ.get('NUM_GPUS', -1)
accelerator = Accelerator(mixed_precision='fp16')
criterion = CrossEntropyLoss()


def multi_core_BERTDATA(filename: str, tokenizer_mc):
    # print("error print if >1")
    temp_set = BERTData(json_path=filename, tokenizer=tokenizer_mc, context_length=1024)
    return temp_set


def multi_core_Labled_BERTDATA(filename: str, tokenizer_mc):
    # print("error print if >1")
    temp_set = BERTDataLabled(json_path=filename, tokenizer=tokenizer_mc, context_length=1024)
    return temp_set


def single_core_BERTDATA(filename: str, tokenizer_sc):
    tmp_set = BERTData_SingleThread(path=filename, tokenizer_pp=tokenizer_sc)
    return tmp_set


def DeBERTa_Tokenizer():
    tokenizerGrab = DebertaTokenizerFast.from_pretrained("microsoft/deberta-v2-xlarge")
    return tokenizerGrab


# DATASET OPTIONS: GLUE IS RECOMMENDED
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
    TrainingChatData = multi_core_BERTDATA('wikitext-103-v1-train.json', tokenizer)
    ValidatingChatData = multi_core_BERTDATA('wikitext-103-v1-validation.json', tokenizer)
    # Create distributed version of the dataset
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(TrainingChatData, batch_size=2, shuffle=True, pin_memory=True)
    ValidatingChatData = DataLoader(ValidatingChatData, batch_size=2, pin_memory=True)

    return TrainingChatData, ValidatingChatData


def open_openWebText():
    if os.path.isfile('openwebtext.json'):
        print('Loading OpenWebText JSON File...')
    else:
        print('Generating OpenWebText JSON File...')
        openwebtext_data = LoadOpenWebText()
        openwebtext_data.save_to_json('openwebtext.json')
    print("Preprocessing...")
    TrainingChatData = multi_core_BERTDATA('openwebtext.json', tokenizer)
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(TrainingChatData, batch_size=16, shuffle=True, pin_memory=True)

    return TrainingChatData


def open_glue():
    if os.path.isfile('glue_train.json'):
        print('Loading GLUE Train JSON File...')
    else:
        print('Generating GLUE Train JSON File...')
        glue_train_data = LoadDataset('glue', 'train')
        glue_train_data.save_to_json('glue_train.json')

    if os.path.isfile('glue_validation.json'):
        print('Loading GLUE Validation JSON File...')
    else:
        print('Generating GLUE Validation JSON File...')
        glue_validation_data = LoadDataset('glue', 'validation')
        glue_validation_data.save_to_json('glue_validation.json')

    print("Preprocessing...")
    TrainingChatData = multi_core_Labled_BERTDATA('glue_train.json', tokenizer)
    ValidatingChatData = multi_core_Labled_BERTDATA('glue_validation.json', tokenizer)

    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(TrainingChatData, batch_size=16, shuffle=True, pin_memory=True)
    ValidatingChatData = DataLoader(ValidatingChatData, batch_size=16, pin_memory=True)

    return TrainingChatData, ValidatingChatData


# Have the model conduct Inferences

def infer(prompt):
    inputs = tokenizer.encode_plus(prompt, return_tensors="pt")
    # noinspection PyTypeChecker
    mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model.to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    mask_word_logits = logits[0, mask_index, :]
    top_5_tokens = torch.topk(mask_word_logits, 5, dim=1).indices[0].tolist()

    probabilities = torch.nn.functional.softmax(mask_word_logits, dim=1)[0]
    top_5_token_probs = probabilities[top_5_tokens].tolist()

    for i, token in enumerate(top_5_tokens):
        word = tokenizer.decode([token])
        probability = top_5_token_probs[i]
        print(f"{word}: {probability}")


def train_and_validate(trainer_data, val_data, model_pass, optimizer, scheduler_pass, epoch_pass):
    model_pass.train()
    val_loss = 0
    num_iterations = 0
    accumulation_steps = 1
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
            val_loss += accelerator.gather(loss).item()

        torch.cuda.empty_cache()
        accelerator.free_memory()

        val_loss /= len(val_data)
        print(f"Test Response for Epoch {epochR}:")
        infer("paris is the [MASK] of France")

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
    mp.set_start_method('fork')
    # tokenizer Declaration and special token Declaration
    tokenizer = DeBERTa_Tokenizer()
    # Model Declaration
    model = DebertaForMaskedLM.from_pretrained("microsoft/deberta-v2-xlarge")
    model.resize_token_embeddings(len(tokenizer))
    # Load Data
    TrainChatData, ValidationChatData = open_glue()
    # Define Optimizer and Scheduler
    optim = Adam(model.parameters(), lr=5e-6)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2, eta_min=0.0001)
    # Accelerate Distributed passthrough
    # noinspection PyTypeChecker
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
        # DeBERTaV2 has a maxVocab of: 128100 tokens
        # According to OpenAI, the conversion rate of character to token is 4:1
        # Cite: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        # Training RunTime
        print("Fine-tuning...")
        train_and_validate(TrainChatData, ValidationChatData, model, optim, scheduler, epoch)
        print("successful fine-tuning...")
        print("Testing Model Training Results With Validation Prompt...")
        for x in range(10):
            infer("paris is the [MASK] of France")
    except KeyboardInterrupt:
        print("aborted")
