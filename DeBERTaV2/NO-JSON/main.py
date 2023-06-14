# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training BERT with glue from HuggingFace
# Backend: Pytorch

from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForMaskedLM, BertTokenizer
import os
import csv
import warnings
from DataSetsForLLM import WikiTextDataset, OpenWebTextDataset, BookCorpusDataset
from tqdm import tqdm
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.cuda
from accelerate import Accelerator
import time


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
    warnings.warn("No validation Set Present, Loading WikiText-103-v1 Validation Set", UserWarning)
    validation_data = WikiTextDataset('validation', tokenizer)
    # Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
    print("Preprocessing...")
    # Create distributed version of the dataset
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collate_fn)
    ValidatingChatData = DataLoader(validation_data, batch_size=8, collate_fn=collate_fn)
    return TrainingChatData, ValidatingChatData


def open_BookCorpus():
    print('Loading Training Data Files...')
    train_data = BookCorpusDataset(tokenizer)
    # Load Wikitext-103-v1 Validation Split and convert it to .json formatting
    print('Loading Validation Data Files...')
    warnings.warn("No validation Set Present, Loading WikiText-103-v1 Validation Set", UserWarning)
    validation_data = WikiTextDataset('validation', tokenizer)
    # Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
    print("Preprocessing...")
    # Create distributed version of the dataset
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collate_fn)
    ValidatingChatData = DataLoader(validation_data, batch_size=8, collate_fn=collate_fn)
    return TrainingChatData, ValidatingChatData


def BERT_Large_Tokenizer():
    tokenizerGrab = BertTokenizer.from_pretrained("bert-large-uncased")
    return tokenizerGrab


def collate_fn(batch):
    input_ids = [item['input_ids'].squeeze() for item in batch]
    attention_masks = [item['attention_mask'].squeeze() for item in batch]
    token_type_ids = [item['token_type_ids'].squeeze() for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True)

    return input_ids, attention_masks, token_type_ids


# Have the model conduct Inferences

def infer(prompt):
    # Encode the input prompt, looking for masked tokens
    inputs = tokenizer.encode_plus(prompt, return_tensors="pt")
    # Locate the masked token(s)
    mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]

    if torch.cuda.is_available():
        # Move everything to the GPU if available
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model.to("cuda")
    with torch.no_grad():
        # Generate output from the model
        outputs = model(**inputs)

    # Retrieve the logits from the output
    logits = outputs.logits
    # Get the logits for the masked word(s)
    mask_word_logits = logits[0, mask_index, :]
    # Find the top 5 predicted tokens and their indices
    top_5_tokens = torch.topk(mask_word_logits, 5, dim=1).indices[0].tolist()
    # Calculate the probabilities of each token prediction
    probabilities = torch.nn.functional.softmax(mask_word_logits, dim=1)[0]
    top_5_token_probs = probabilities[top_5_tokens].tolist()
    # Print out the predicted words and their probabilities
    for i, token in enumerate(top_5_tokens):
        word = tokenizer.decode([token])
        probability = top_5_token_probs[i]
        print(f"{word}: {probability}")


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
            inputs, attention_masks, token_type_ids = batch
            batch_size = len(inputs)
            optimizer.zero_grad()
            outputs = model_pass(input_ids=inputs, token_type_ids=token_type_ids, attention_mask=attention_masks)
            loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), inputs.view(-1))
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
            inputs, attention_masks, token_type_ids = batch
            outputs = model_pass(input_ids=inputs, token_type_ids=token_type_ids, attention_mask=attention_masks)
            loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), inputs.view(-1))
            val_loss += accelerator.gather(loss)

        torch.cuda.empty_cache()
        accelerator.free_memory()

        val_loss /= len(val_data)
        print(f"Test Response for Epoch: {epochR}")
        infer("Albert Einstein was best known for his [MASK] theory of relativity.")

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
    tokenizer = BERT_Large_Tokenizer()
    # Model Declaration
    model = BertForMaskedLM.from_pretrained("bert-large-uncased")
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
        Test_Prompts = ["Albert Einstein was best known for his [MASK] theory of relativity.",
                        "The Eiffel Tower is located in [MASK].",
                        "The largest organ in the human body is the [MASK].",
                        "The capital of the United States is [MASK].",
                        "Apple Inc. was co-founded by Steve [MASK].",
                        "The [MASK] is the closest star to Earth.",
                        "J.K. Rowling is famous for writing the [MASK] series."]
        for x in Test_Prompts:
            print(f"Test: {x}, Prompt: {Test_Prompts[x]}, Results: ")
            infer(Test_Prompts[x])

    except KeyboardInterrupt:
        print("Aborted by the User")
