# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training GPT2 with Wikitext103-v1 from HuggingFace
# Backend: Pytorch
# High level Imports
import os
import csv
import time
import psutil
import sys
import threading
# Pytorch Imports
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW, lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.cuda
# Hugging Face Imports
from transformers import GPT2LMHeadModel, AutoTokenizer
from DataSetsForLLM import WikiTextDataset, OpenWebTextDataset
from accelerate import Accelerator
# Progress Bar Imports
from tqdm import tqdm

# Environment Defined training Parameters
Ben_batch_size = int(os.environ.get('BATCH_TRAIN_SIZE')) if os.environ.get('BATCH_TRAIN_SIZE') is not None else 8
model_name = os.environ.get('MODEL_NAME') if not None else "gpt2-large"
precision_type = os.environ.get('PRECISION_TRAIN') if not None else "fp16"
tokenizer_type = os.environ.get('MODEL_TOKENIZER') if not None else "gpt2-large"
task_type = os.environ.get("TASK_TYPE") if not None else "finetune"
benchmark_time_interval = int(os.environ.get('BM_INTERVAL')) if os.environ.get('BM_INTERVAL') is not None else 1
# Define Device for Training
accelerator = Accelerator(mixed_precision=precision_type)
criterion = CrossEntropyLoss()


def open_wikiText():
    """
    Parameters
    None

    Returns
    TrainingChatData : DataLoader
        DataLoader object containing the training data for the model.
    ValidatingChatData : DataLoader
        DataLoader object containing the validation data for the model.

    Description This function loads and preprocesses the training and validation data files for a chatbot model using
    the WikiText dataset. It performs the following steps: Loads the training data files. Loads the validation data
    files. Preprocesses the data. Creates distributed versions of the datasets. Returns the DataLoader objects for
    training and validation data.
    """
    print('Loading Training Data Files...')
    train_data = WikiTextDataset('train', tokenizer)

    # Load Wikitext-103-v1 Validation Split and convert it to .json formatting
    print('Loading Validation Data Files...')
    validation_data = WikiTextDataset('validation', tokenizer)
    # Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
    print("Preprocessing...")
    # Create distributed version of the dataset
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(train_data, batch_size=Ben_batch_size, shuffle=True, collate_fn=collate_fn)
    ValidatingChatData = DataLoader(validation_data, batch_size=Ben_batch_size, collate_fn=collate_fn)
    return TrainingChatData, ValidatingChatData


def open_WebText():
    """
    Parameters
    None

    Returns
    TrainingChatData : DataLoader
        DataLoader object containing the training data for the model.
    ValidatingChatData : DataLoader
        DataLoader object containing the validation data for the model.

    Description This function loads and preprocesses the training and validation data files for a chatbot model using
    the OpenWebText and WikiText datasets. It performs the following steps:
        Loads the training data files from OpenWebTextDataset.
        Loads the validation data files from WikiTextDataset.
        Preprocesses the data.
        Creates distributed versions of the datasets.
        Returns the DataLoader objects for training and validation data.
    """
    print('Loading Training Data Files...')
    train_data = OpenWebTextDataset(tokenizer, 'train')
    # Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
    print("Preprocessing...")
    # Create distributed version of the dataset
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(train_data, batch_size=Ben_batch_size, shuffle=True, collate_fn=collate_fn)
    return TrainingChatData


def GPT2_Tokenizer():
    """
    Parameters
    None

    Returns
    tokenizerGrab : GPT2Tokenizer
        GPT2Tokenizer object for tokenizing text using the GPT-2 large model.

    Description
    This function retrieves the GPT2Tokenizer from the 'gpt2-large' pretrained model. It performs the following steps:
        Retrieves the tokenizer using AutoTokenizer.from_pretrained().
        Sets the padding token to '<pad>'.
        Sets the end-of-sequence token to '<eos>'.
        Sets the beginning-of-sequence token to '<bos>'.
        Returns the GPT2Tokenizer object.
    """

    tokenizerGrab = AutoTokenizer.from_pretrained(tokenizer_type, use_fast=True)
    tokenizerGrab.pad_token = '<pad>'
    tokenizerGrab.eos_token = '<eos>'
    tokenizerGrab.bos_token = '<bos>'
    return tokenizerGrab


def collate_fn(batch):
    """
    Parameters
    batch : List
        A list of dictionaries, where each dictionary represents
        a batch item with 'input_ids' and 'attention_mask' keys.

    Returns
    input_ids : Tensor
        Padded tensor of shape (batch_size, max_sequence_length) containing the input IDs for each batch item.
    attention_masks : Tensor
        Padded tensor of shape (batch_size, max_sequence_length) containing the attention masks for each batch item.

    Description This function is a collate function used in data loading for creating batches of data. It takes a
    batch of samples and performs the following steps:
        Extracts the 'input_ids' from each dictionary item in the batch.
        Extracts the 'attention_mask' from each dictionary item in the batch.
        Pads the 'input_ids' sequences to have the same length within the batch using the pad_sequence function.
        Pads the 'attention_masks' sequences to have the same length within the batch using the pad_sequence function.
        Returns the padded 'input_ids' and 'attention_masks' as tensors.
    """
    input_ids = [item['input_ids'].squeeze() for item in batch]
    attention_masks = [item['attention_mask'].squeeze() for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)

    return input_ids, attention_masks


# Have the model conduct Inferences

def infer(prompt, modelI, tokenizerI):
    """
    Parameters
    prompt : str
        The input prompt for generating text.

    Returns
    generated_text : str
        The generated text based on the input prompt.

    Description This function performs text generation using a pre-trained language model. It takes an input prompt
    and generates text based on the prompt using the following steps:

    Encodes the input prompt using the tokenizer, returning a tensor representation of the input.
    Creates an attention mask tensor of ones with the same shape as the input tensor.
    If a CUDA-enabled GPU is available,
    moves the input tensor and attention mask tensor to the GPU and sets the model to use the GPU.
    Generates text
    using the model's generate method, passing the input tensor, attention mask, and the ID for the end-of-sentence
    token as the padding token ID.
    Decodes the generated output tensor into human-readable text, skipping any special tokens.
    Returns the generated text.
    """
    inputs = tokenizerI.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
        attention_mask = attention_mask.to("cuda")
        modelI.to("cuda")
    outputs = modelI.generate(inputs,
                              attention_mask=attention_mask,
                              pad_token_id=tokenizerI.eos_token_id,
                              max_length=150)
    generated_text = tokenizerI.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def train_model(trainer_data, model_pass, optimizer, scheduler_pass, epoch_pass):
    """
    Parameters
    ----------
    trainer_data : DataLoader
        DataLoader object containing the training data for the model.
    model_pass : transformer.GPT2LMHeadModel
        The model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer used for training the model.
    scheduler_pass : torch.optim.lr_scheduler._LRScheduler
        The scheduler used for adjusting the learning rate during training.
    epoch_pass : int
        The total number of epochs to train the model.

    Returns
    -------
    None

    Description
    -----------
    This function performs the training process for a given model using the specified
    training data. It follows the following steps:

    1. Sets the model to training mode.
    2. Initializes variables for tracking training loss, number of iterations, and batch size.
    3. Sets the default tensor type to "torch.FloatTensor".
    4. Iterates through each epoch.
    5. Within each epoch, iterates through each batch of the training data.
    6. If the batch is None (empty), skips to the next iteration which accounts for the last batch being empty.
    7. Retrieves inputs and targets from the batch.
    8. Zeroes the gradients of the optimizer.
    9. Passes the inputs through the model to get the outputs.
    10. Calculates the loss using the specified criterion.
    11. Performs backward propagation of gradients using the accelerator (if available).
    12. Updates the model parameters based on the gradients and optimizer's update rule.
    13. Adjusts the learning rate using the scheduler.
    14. Performs gradient accumulation if the number of iterations reaches the specified accumulation steps.
    15. Frees memory using the accelerator.
    16. Calculates the epoch time and throughput.
    17. Saves the model's state dictionary to a file.
    18. Writes the training loss, epoch time, and throughput to a CSV file.

    Note
    ----
    The validation aspect of the original function has been removed in this version.

    """
    file_path = f'training_results.csv'
    # Initialize Model mode and High level Values
    model_pass.train()
    max_grad_norm = 1
    num_iterations = 0
    accumulation_steps = 1
    torch.set_default_tensor_type("torch.FloatTensor")

    # Write the header to the CSV file
    with open(file_path, mode='w', newline='') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(['Epoch',
                         'Batch',
                         'Training Loss',
                         'Time',
                         'Throughput (Seq/sec)',
                         'Disk Read IOPS',
                         'Disk Write IOPS'])
    try:
        for epochR in range(epoch_pass):
            # Training loop
            start = time.time()
            training_losses = []  # Initialize list to store training losses
            disk_read_count = 0  # Initialize disk read counter
            disk_write_count = 0  # Initialize disk write counter
            with open(file_path, mode='a', newline='') as results_file:
                writer = csv.writer(results_file)
                for batch_num, batch in enumerate(
                        tqdm(trainer_data, desc=f'Training Epoch {epochR}, Batch', leave=True)):
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

                    training_losses.append(loss.item())  # Store the training loss

                    # Gradient Accumulation
                    if num_iterations == accumulation_steps:
                        # Gradient clipping
                        clip_grad_norm_(model_pass.parameters(), max_grad_norm)
                        optimizer.step()
                        scheduler_pass.step(loss)
                        num_iterations = 0

                    # Update disk IOPS counters
                    disk_io_counters = psutil.disk_io_counters()
                    disk_read_count += disk_io_counters.read_count
                    disk_write_count += disk_io_counters.write_count

                    # Calculate Batch time
                    end = time.time()
                    batch_time = end - start

                    # Calculate throughput of the system for each batch
                    throughput = batch_size * (batch_num + 1) / batch_time

                    # Calculate disk read and write IOPS for each batch
                    disk_read_iops = disk_read_count / batch_time
                    disk_write_iops = disk_write_count / batch_time

                    # Write the results for each batch
                    writer.writerow(
                        [epochR, batch_num, training_losses[-1], batch_time, throughput, disk_read_iops,
                         disk_write_iops])

            accelerator.free_memory()
    except Exception as e:
        # Handle exception and log error message
        print("Error:", e)
        # Close the CSV file properly
        sys.exit(1)
    except KeyboardInterrupt:
        # Handle exception and log error message
        print("Training Loop Aborted")
        # Close the CSV file properly
        sys.exit(1)


def monitor_system_utilization(interval):
    """
    Monitor System Utilization

    This function monitors the CPU utilization, RAM utilization, and core utilization of the system until a keyboard
    interrupt (Ctrl+C) is triggered.

    Parameters
    ----------
    interval : float
        The interval (in seconds) at which the utilization is measured.

    Returns
    -------
    None

    """
    file_path = 'CPU_RAM_Utilization.csv'
    with open(file_path, mode='a', newline='') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(['Core Time', 'CPU Utilization', 'Thread Count', 'RAM Utilization (%)', 'RAM Utilization (MB)'])

    try:
        core_time = 0
        while True:
            # Get CPU utilization
            cpu_percent = psutil.cpu_percent(interval=interval)

            # Get RAM utilization
            ram = psutil.virtual_memory()
            ram_percent = ram.percent
            ram_mb = ram.used / (1024 * 1024)  # Convert bytes to megabytes (MB)

            # Get thread count
            threads = threading.active_count()

            core_time += interval
            with open(file_path, mode='a', newline='') as results_file:
                writer = csv.writer(results_file)
                writer.writerow([core_time, cpu_percent, threads, ram_percent, ram_mb])

            # Sleep for the specified interval
            time.sleep(interval)

    except KeyboardInterrupt:
        # Handle exception and log error message
        print("Training Loop Aborted")
        # Close the CSV file properly
        sys.exit(1)


def pretrain_model(training_data_loader, modelPM, optimizer, schedulerPM, total_epochs):
    """
    Pretrain Model

    This function performs the pretraining process for a given model using the specified
    training data. It follows the following steps:

    1. Sets the model to training mode.
    2. Initializes variables for tracking training loss, number of iterations, and batch size.
    3. Sets the default tensor type to "torch.FloatTensor".
    4. Iterates through each epoch.
    5. Within each epoch, iterates through each batch of the training data.
    6. If the batch is None (empty), skips to the next iteration which accounts for the last batch being empty.
    7. Retrieves inputs and targets from the batch.
    8. Zeroes the gradients of the optimizer.
    9. Passes the inputs through the model to get the outputs.
    10. Calculates the loss using the specified criterion.
    11. Performs backward propagation of gradients using the accelerator (if available).
    12. Updates the model parameters based on the gradients and optimizer's update rule.
    13. Adjusts the learning rate using the scheduler.
    14. Performs gradient accumulation if the number of iterations reaches the specified accumulation steps.
    15. Frees memory using the accelerator.
    16. Calculates the epoch time and throughput.
    17. Saves the model's state dictionary to a file.
    18. Writes the training loss, epoch time, and throughput to a CSV file.

    Parameters
    ----------
    training_data_loader : DataLoader
        DataLoader object containing the training data for the model.
    modelPM : transformer.GPT2LMHeadModel
        The model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer used for training the model.
    schedulerPM : torch.optim.lr_scheduler._LRScheduler
        The scheduler used for adjusting the learning rate during training.
    total_epochs : int
        The total number of epochs to pretrain the model.

    Returns
    -------
    None

    Note
    ----
    The validation aspect of the original function has been removed in this version.

    """
    file_path = f'training_results.csv'
    # Initialize Model mode and High level Values
    modelPM.train()
    max_grad_norm = 1
    num_iterations = 0
    accumulation_steps = 1
    torch.set_default_tensor_type("torch.FloatTensor")

    # Write the header to the CSV file
    with open(file_path, mode='w', newline='') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(['Epoch',
                         'Batch',
                         'Training Loss',
                         'Time',
                         'Throughput (Seq/sec)',
                         'Disk Read IOPS',
                         'Disk Write IOPS'])

    try:
        for epochR in range(total_epochs):
            # Training loop
            start = time.time()
            training_losses = []  # Initialize list to store training losses
            disk_read_count = 0  # Initialize disk read counter
            disk_write_count = 0  # Initialize disk write counter
            with open(file_path, mode='a', newline='') as results_file:
                writer = csv.writer(results_file)
                for batch_num, batch in enumerate(
                        tqdm(training_data_loader, desc=f'Epoch {epochR}, Batch', leave=True)):
                    if batch is None:
                        # Skip the last empty batch (As the multicore Encoder returns NoneType for last index)
                        continue
                    inputs, targets = batch
                    batch_size = len(inputs)
                    optimizer.zero_grad()
                    outputs = modelPM(inputs)
                    loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), targets.view(-1))
                    accelerator.backward(loss)
                    num_iterations += 1

                    training_losses.append(loss.item())  # Store the training loss

                    # Gradient Accumulation
                    if num_iterations == accumulation_steps:
                        # Gradient clipping
                        clip_grad_norm_(modelPM.parameters(), max_grad_norm)
                        optimizer.step()
                        schedulerPM.step(loss)
                        num_iterations = 0

                    # Update disk IOPS counters
                    disk_io_counters = psutil.disk_io_counters()
                    disk_read_count += disk_io_counters.read_count
                    disk_write_count += disk_io_counters.write_count

                    # Calculate Batch time
                    end = time.time()
                    batch_time = end - start

                    # Calculate throughput of the system for each batch
                    throughput = batch_size * (batch_num + 1) / batch_time

                    # Calculate disk read and write IOPS for each batch
                    disk_read_iops = disk_read_count / batch_time
                    disk_write_iops = disk_write_count / batch_time

                    # Write the results for each batch
                    writer.writerow(
                        [epochR, batch_num, training_losses[-1], batch_time, throughput, disk_read_iops,
                         disk_write_iops])

            accelerator.free_memory()
    except Exception as e:
        # Handle exception and log error message
        print("Error:", e)
        # Close the CSV file properly
        sys.exit(1)
    except KeyboardInterrupt:
        # Handle exception and log error message
        print("Training Loop Aborted")
        # Close the CSV file properly
        sys.exit(1)


if __name__ == '__main__':
    # Start monitoring system utilization
    Measure_interval = benchmark_time_interval  # Interval between measurements (in seconds)
    utilization_monitor = threading.Thread(target=monitor_system_utilization, args=(Measure_interval,))
    utilization_monitor.start()
    # tokenizer Declaration and special token Declaration
    tokenizer = GPT2_Tokenizer()
    # Model Declaration
    model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    model.resize_token_embeddings(len(tokenizer))
    # Load Data
    TrainChatData = open_WebText()
    # TrainChatData, ValidationChatData = open_OpenWebText()
    # Define Optimizer and Scheduler
    optim = AdamW(model.parameters(), lr=5e-6)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2, eta_min=0.0001)
    # Accelerate Distributed passthrough
    model, optim, scheduler, TrainChatData = accelerator.prepare(model,
                                                                 optim,
                                                                 scheduler,
                                                                 TrainChatData,
                                                                 )
    try:
        # Call Training Function (Will write a CSV file)
        epoch = int(os.environ.get('NUM_EPOCHS')) if os.environ.get('NUM_EPOCHS') is not None else 1
        # Set Token length per Text Entry
        # (Entries Longer than specified number will be truncated and Entries Shorter will be Padded with empty tokens)
        # GPT2 has a max length of 1024 tokens
        # According to OpenAI, the conversion rate of character to token is 4:1
        # Cite: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        # Training RunTime
        print("Fine-tuning...")
        train_model(TrainChatData, model, optim, scheduler, epoch)
        print("successful fine-tuning...")
        print("Testing Model Training Results With Validation Prompt...")
        torch.cuda.empty_cache()
        accelerator.free_memory()
        # Initialize an empty list to store the model generation strings
        generated_strings = []

        for x in range(10):
            ModelGeneration = infer("Albert Einstein was ", model, tokenizer)
            print(ModelGeneration)
            generated_strings.append(ModelGeneration)

        # Write the generated strings to a CSV file
        output_file_path = 'model_output.csv'
        with open(output_file_path, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['Generated Strings'])
            for string in generated_strings:
                csv_writer.writerow([string])
        print("Model output written to CSV file:", output_file_path)

    except KeyboardInterrupt:
        print("Aborted by the User")
