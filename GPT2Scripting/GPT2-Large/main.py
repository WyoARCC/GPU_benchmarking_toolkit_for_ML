# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training BERT with glue from HuggingFace
# Backend: Pytorch

from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForMaskedLM, AutoTokenizer
import os
import csv
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
    train_data = WikiTextDataset(tokenizer=tokenizer, split='train')
    # Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
    print("Preprocessing...")
    # Create distributed version of the dataset
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    return TrainingChatData


def open_WebText():
    print('Loading Training Data Files...')
    train_data = OpenWebTextDataset(tokenizer=tokenizer, split='train')
    # Load Wikitext-103-v1 Validation Split and convert it to .json formatting
    # Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
    print("Preprocessing...")
    # Create distributed version of the dataset
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    return TrainingChatData


def open_BookCorpus():
    print('Loading Training Data Files...')
    train_data = BookCorpusDataset(tokenizer=tokenizer, split='train')
    # Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
    print("Preprocessing...")
    # Create distributed version of the dataset
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    return TrainingChatData


def BERT_Large_Tokenizer():
    tokenizerGrab = AutoTokenizer.from_pretrained("bert-large-cased")
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

def infer(prompt, model, tokenizer):
    """
    Parameters
    ----------
    prompt : str
        The input prompt for generating predictions.

    Description
    -----------
    This function performs text inference using a pre-trained language model. It takes an
    input prompt and generates predictions for masked tokens in the prompt using the following steps:

    1. Encode the input prompt using the tokenizers `encode_plus` method, returning a dictionary of input tensors.
    2. Locate the masked token(s) in the input tensor.
    3. If a CUDA-enabled GPU is available, move the input tensors and the model to the GPU.
    4. Disable gradient calculations by wrapping the following code block with `torch.no_grad()`.
    5. Generate output from the model by passing the input tensors as keyword arguments.
    6. Retrieve the logits from the output.
    7. Get the logits for the masked word(s) by indexing the logits tensor with the mask indices.
    8. Find the top 5 predicted tokens and their indices based on the highest logits.
    9. Calculate the probabilities of each token prediction by applying softmax to the mask word logits.
    10. Convert the top 5 token indices and their probabilities to lists.
    11. Print out the predicted words and their probabilities.

    Note: The `tokenizer` and `model` variables used in this function need to be defined and available
    in the current scope.

    """
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


def train_and_validate(trainer_data, model_pass, optimizer, scheduler_pass, epoch_pass):
    """
    train_and_validate

    This function is responsible for training and validating a model using the provided data.
    It takes several parameters as input and performs the following steps:

    Parameters
    -------------

    trainer_data (iterable):
        The training data for the model.
    model_pass (torch.nn.Module):
        The model to be trained and validated.
    optimizer (torch.optim.Optimizer):
        The optimizer used for updating the model's parameters.
    scheduler_pass (torch.optim.lr_scheduler._LRScheduler):
        The learning rate scheduler used for adjusting the learning rate during training.
    epoch_pass (int):
        The number of training epochs.

    Description
    -------------

    1. Initialize the file path and name for saving the training results.
    2. Set the model to training mode.
    3. Set the maximum gradient norm to 1.
    4. Initialize variables for tracking the number of iterations and batch size.
    5. Set the default tensor type to "torch.FloatTensor".
    6. Start the training loop for the specified number of
    epochs.
    7. Start the timer for tracking the epoch time.
    8. Initialize an empty list to store the training losses
    for each batch.
    9. Iterate over the training data batches using tqdm for progress tracking.
    10. Skip the last empty batch (if any).
    11. Extract inputs, attention masks, and token type IDs from the batch.
    12. Zero the gradients of the optimizer.
    13. Forward pass the inputs through the model to obtain the outputs.
    14. Calculate the loss between the outputs and the inputs.
    15. Perform backward propagation of gradients through the model.
    16.
    Increment the number of iterations.
    17. Append the current batch's loss to the training_losses list.
    18. Check if
    the number of iterations equals the accumulation steps.
    19. If so, clip the gradients to prevent exploding
    gradients.
    20. Update the model's parameters using the optimizer.
    21. Adjust the learning rate using the
    scheduler.
    22. Reset the number of iterations to 0.
    23. Free the memory used by the accelerator (if applicable).
    24. End the timer for the epoch time.
    25. Calculate the throughput of the system.
    26. Save the model's state dictionary to a file with the epoch number in the name.
    27. Check if the training_results.csv file exists.
    28. If not, create the file and write the header.
    29. Open the file in append mode and write the epoch number,
    batch number, training loss, epoch time, and throughput for each batch.
    30. Free the memory used by the
    accelerator (if applicable).

    The function performs the training and validation process for the specified number of epochs, saving the model's
    state after each epoch and recording the training losses, epoch time, and throughput in a CSV file named
    "training_results.csv".

    """
    # Specify the file path and name
    file_path = 'training_results.csv'
    model_pass.train()
    max_grad_norm = 1
    num_iterations = 0
    accumulation_steps = 1
    torch.set_default_tensor_type("torch.FloatTensor")
    for epochR in range(epoch_pass):
        # Training loop
        start = time.time()
        batch_size = 0
        training_losses = []  # Initialize list to store training losses
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
            # Training loss list
            training_losses.append(loss.item())  # Store the training loss

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

        # Calculate Throughput of the System
        throughput = batch_size * len(trainer_data) / epochTime

        # Save model and write training loss to CSV file
        accelerator.save(model_pass.state_dict(), f"model_state_{epochR}.pt")
        # Check if the file already exists
        if not os.path.exists(file_path):
            # Create the file and write the header
            with open(file_path, mode='w', newline='') as results_file:
                writer = csv.writer(results_file)
                writer.writerow(['Epoch', 'Batch', 'Training Loss', 'Time', 'Throughput'])

        # Open the file in append mode and write the results
        with open(file_path, mode='a', newline='') as results_file:
            writer = csv.writer(results_file)
            for batch_num, training_loss in enumerate(training_losses):
                writer.writerow([epochR, batch_num, training_loss, epochTime, throughput])

    accelerator.free_memory()


if __name__ == '__main__':
    # tokenizer Declaration and special token Declaration
    tokenizer = BERT_Large_Tokenizer()
    # Model Declaration
    model = BertForMaskedLM.from_pretrained("bert-large-cased")
    # Load Data
    TrainChatData = open_WebText()
    # TrainChatData, ValidationChatData = open_OpenWebText()
    # Define Optimizer and Scheduler
    optim = Adam(model.parameters(), lr=5e-6)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2, eta_min=0.0001)
    # Accelerate Distributed passthrough
    model, optim, scheduler, TrainChatData = accelerator.prepare(model,
                                                                 optim,
                                                                 scheduler,
                                                                 TrainChatData,
                                                                 )
    try:
        # Call Training Function (Will write a CSV file)
        epoch = 2
        # Set Token length per Text Entry
        # (Entries Longer than specified number will be truncated and Entries Shorter will be Padded)
        # GPT2 has a max length of 1024 tokens
        # According to OpenAI, the conversion rate of character to token is 4:1
        # Cite: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        os.environ['max_tok_length'] = str(256)
        # Training RunTime
        print("Fine-tuning...")
        train_and_validate(TrainChatData, model, optim, scheduler, epoch)
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
            infer(Test_Prompts[x], model, tokenizer)

    except KeyboardInterrupt:
        print("Aborted by the User")
