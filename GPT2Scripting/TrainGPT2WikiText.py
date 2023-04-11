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


def preprocess_function(examples):
    tokenizer_pp = GPT2TokenizerFast.from_pretrained("gpt2-xl")
    max_length = 1024  # replace with an appropriate value based on your requirements
    text_list = [text.strip().replace("\n", "<eos>") for text in examples["text"]]
    encoded_inputs = tokenizer_pp(text_list, truncation=True, max_length=max_length, padding="max_length")
    return {"input_ids": encoded_inputs["input_ids"],
            "token_type_ids": encoded_inputs["token_type_ids"],
            "attention_mask": encoded_inputs["attention_mask"]}


def wiki_dataset(core_count: int):
    # loads Wikitext dataset and tokenizes it using the GPT2Tokenizer

    # using Huggingface dataset library
    train_datasets = load_dataset("wikitext", "wikitext-103-v1", split="train")
    train_datasets.save_to_disk('training_wikitext_dataset')

    eval_datasets = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    eval_datasets.save_to_disk('validation_wikitext_dataset')
    # Preprocess the datasets
    ordered_train_dataset = train_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=core_count,
        remove_columns=["text"],
    )
    ordered_validation_dataset = eval_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=core_count,
        remove_columns=["text"]
    )
    # Format to Pytorch Requirements
    processed_train_dataset = ordered_train_dataset.set_format(type="torch",
                                                               columns=["input_ids", "token_type_ids",
                                                                        "attention_mask"])
    processed_eval_dataset = ordered_validation_dataset.set_format(type="torch",
                                                                   columns=["input_ids", "token_type_ids",
                                                                            "attention_mask"])
    return processed_train_dataset, processed_eval_dataset


if __name__ == '__main__':
    # KEEP TRACK OF THESE WHEN DEVELOPING UI/UX
    num_epochs = 4
    batch_size = 8
    accumulation_steps = 32
    tokenization_core_count = 10
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")
    # model declaration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_ctx=1024,
        n_embd=1600,
        n_layer=48,
        n_head=25,
        dropout=0.1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True
    )
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl", config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    model.to(device)

    # TRAINING MODE
    model.train()
    # load preprocessed Wikitext
    Wiki_train, Wiki_validate = wiki_dataset(tokenization_core_count)
    # set data loaders
    train_dataloader = torch.utils.data.DataLoader(Wiki_train, batch_size=batch_size, shuffle=True)
    eval_dataloader = torch.utils.data.DataLoader(Wiki_validate, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", unit="batch")
        for step, batch_list in enumerate(progress_bar):
            for batch in batch_list:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["input_ids"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
                loss = loss / accumulation_steps
                loss.backward()

                total_loss += loss.item()

                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    model.zero_grad()

            gpu_memory = torch.cuda.max_memory_allocated(device=device)
            ram_memory = psutil.Process().memory_info().rss / 1024 ** 2  # in MB
            progress_bar.set_postfix(
                {"batch_loss": loss.item(), "avg_loss": total_loss / ((step + 1) * len(batch_list)),
                 "gpu_memory": gpu_memory / 1024 ** 2, "ram_memory": ram_memory})

        # SET EVALUATE MODE and Evaluate the model after each epoch
        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["input_ids"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))

                total_eval_loss += loss.item()

        avg_eval_loss = total_eval_loss / len(eval_dataloader)
        print(
            f"Epoch {epoch + 1}: avg_train_loss={total_loss / len(train_dataloader):.4f},"
            f" avg_eval_loss={avg_eval_loss:.4f}")

