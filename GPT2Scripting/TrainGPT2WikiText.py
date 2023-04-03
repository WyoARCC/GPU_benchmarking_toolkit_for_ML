# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training GPT2 with Wikitext103-v1 from HuggingFace
# Backend: Pytorch

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from transformers import GPT2Model, GPT2Tokenizer, DataCollatorWithPadding, AdamW, get_scheduler
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
# progress bar
from tqdm.auto import tqdm
import evaluate


checkpoint = 'gpt2-xl'
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)


def wiki_dataset():
    # loads Wikitext dataset and tokenizes it using the GPT2Tokenizer
    # using Huggingface dataset library
    raw_datasets = load_dataset("wikitext", "wikitext-103-v1")
    # 🤗 Tokenizers library already uses multiple threads to tokenize the samples we have to define a collate
    # function that will apply the correct amount of padding to the items of the dataset we want to batch together.
    # Fortunately, the 🤗 Transformers library provides us with such a function via DataCollatorWithPadding. It takes
    # a tokenizer when you instantiate it (to know which padding token to use, and whether the model expects padding
    # to be on the left or on the right of the inputs) and will do everything you need:
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # Remove columns: with values "sentence1", "sentence2", "idx" and set tensor format to Pytorch
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    return tokenized_datasets, data_collator


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)


if __name__ == '__main__':
    # print(wiki_dataset())
    # load preprocessed Wikitext and collator
    Wiki_RAW, Collator, = wiki_dataset()
    # Define Data Loaders
    train_dataloader = DataLoader(
        Wiki_RAW["train"], shuffle=True, batch_size=8, collate_fn=Collator
    )
    eval_dataloader = DataLoader(
        Wiki_RAW["validation"], shuffle=True, batch_size=8, collate_fn=Collator
    )
    # To VALIDATE data processing uncomment below:
    # for batch in train_dataloader:
    #    break
    # {k: v.shape for k, v in batch.items()}

    # model declaration
    model = GPT2Model.from_pretrained('gpt2-xl')
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Cuda Hardware Initialization (multiGPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # epoch declaration and initializing training loop parameters
    num_epochs = 5
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # TRAINING LOOP: MAIN EVENT ONLY DEPLOY WHEN READY
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # Evaluation LOOP: MAIN EVENT ONLY DEPLOY WHEN READY
    metric = evaluate.load("wikitext", "wikitext-103-v1")
    model.eval()
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["text"])

    metric.compute()
