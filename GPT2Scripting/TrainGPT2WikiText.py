# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training GPT2 with Wikitext103-v1 from HuggingFace
# Backend: Pytorch

from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel
import torch
from torch.utils.data import DataLoader
import psutil
from datasets import load_dataset
# progress bar
from tqdm.auto import tqdm

max_length = 1024  # maximum length of input sequence
model_config = GPT2Config.from_pretrained("gpt2-xl", output_hidden_states=True)
model = GPT2LMHeadModel.from_pretrained("gpt2-xl", config=model_config)


def preprocess_function(examples):  # THIS SHOULD WORK NOW
    tokenizer_pp = GPT2TokenizerFast.from_pretrained("gpt2-xl")
    for i in range(len(examples["text"])):
        examples["text"][i] = examples["text"][i].strip().replace("\n", "<eos>")
    return tokenizer_pp(examples["text"])


def wiki_dataset(core_count: int):
    # loads Wikitext dataset and tokenizes it using the GPT2Tokenizer

    # using Huggingface dataset library
    train_datasets = load_dataset("wikitext", "wikitext-103-v1", split="train", num_proc=core_count)
    train_datasets.save_to_disk('training_wikitext_dataset')

    eval_datasets = load_dataset("wikitext", "wikitext-103-v1", split="validation", num_proc=core_count)
    eval_datasets.save_to_disk('validation_wikitext_dataset')
    # Preprocess the datasets
    ordered_train_dataset = train_datasets.map(
        preprocess_function,
        batched=True,
    )
    ordered_validation_dataset = eval_datasets.map(
        preprocess_function,
        batched=True,
    )
    # Format to Pytorch Requirements
    processed_train_dataset = ordered_train_dataset.format(type="torch",
                                                           columns=["input_ids", "token_type_ids",
                                                                    "attention_mask"])
    processed_eval_dataset = ordered_validation_dataset.format(type="torch",
                                                               columns=["input_ids", "token_type_ids",
                                                                        "attention_mask"])
    return processed_train_dataset, processed_eval_dataset


if __name__ == '__main__':
    # KEEP TRACK OF THESE WHEN DEVELOPING UI/UX
    num_epochs = 4
    batch_size = 1
    accumulation_steps = 128 // batch_size
    tokenization_core_count = 10
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")
    # load preprocessed Wikitext
    Wiki_train, Wiki_validate = wiki_dataset(tokenization_core_count)

    # model declaration
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    # hardware declaration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # data loaders
    train_dataloader = torch.utils.data.DataLoader(Wiki_train, batch_size=batch_size, shuffle=True, )
    eval_dataloader = torch.utils.data.DataLoader(Wiki_validate, batch_size=batch_size, shuffle=False)

    # TRAINING LOOP: MAIN EVENT ONLY DEPLOY WHEN READY

    model.train()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", unit="batch")
        for step, batch_list in enumerate(train_dataloader):
            for batch in batch_list:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["input_ids"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
                loss.backward()

                total_loss += loss.item()

                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    model.zero_grad()
                progress_bar.set_postfix(
                    {"batch_loss": loss.item(), "avg_loss": total_loss / ((step + 1) * len(batch_list))})
            gpu_memory = torch.cuda.max_memory_allocated(device=device)
            ram_memory = psutil.Process().memory_info().rss / 1024 ** 2  # in MB
            print(f"GPU memory = {gpu_memory / 1024 ** 2:.2f} GB, RAM memory = {ram_memory:.2f} MB")

        avg_train_loss = total_loss / len(train_dataloader)
        model.eval()
        total_eval_loss = 0
        for batch in eval_dataloader:
            with torch.no_grad():
                input_ids = batch["input_ids"].squeeze().to(device)
                attention_mask = batch["attention_mask"].squeeze().to(device)
                labels = batch["input_ids"].squeeze().to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
                total_eval_loss += loss.item()

        avg_eval_loss = total_eval_loss / len(eval_dataloader)

        print(f"Epoch {epoch + 1}:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Eval Loss: {avg_eval_loss:.4f}")
