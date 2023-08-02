# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training BERT with glue from HuggingFace
# Backend: Pytorch
import sys
import threading

import psutil
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForMaskedLM, AutoTokenizer
import os
import csv
from DataSetsForLLM import WikiTextDataset, OpenWebTextDataset, BookCorpusDataset, RedPajamaDataset, PileDataset, \
    StarCoderDataset
from tqdm import tqdm
from torch.optim import AdamW, lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.cuda
from accelerate import Accelerator
import time

# /pfs/tc1/project/arcc-students/tlimato/BERT_SHDev/BERT_Testing

# Environment Defined training Parameters
Ben_batch_size = int(os.environ.get('BATCH_TRAIN_SIZE')) if os.environ.get('BATCH_TRAIN_SIZE') is not None else 32
model_name = os.environ.get('MODEL_NAME') if not None else "bert-large-uncased"
precision_type = os.environ.get('PRECISION_TRAIN') if not None else "fp16"
tokenizer_type = os.environ.get('MODEL_TOKENIZER') if not None else "bert-large-uncased"
task_type = os.environ.get("TASK_TYPE") if not None else "finetune"
benchmark_time_interval = int(os.environ.get('BM_INTERVAL')) if os.environ.get('BM_Interval') is not None else 1
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
    # Create distributed version of the dataset
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(train_data, batch_size=Ben_batch_size, shuffle=True, collate_fn=collate_fn)
    return TrainingChatData


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


def open_RedPajama():
    """
    Parameters
    None

    Returns
    TrainingChatData : DataLoader
        DataLoader object containing the training data for the model.

    Description This function loads and preprocesses the training data files for a chatbot model using
    the RedPajama-Data-1T-Sample dataset. It performs the following steps:
        Loads the training data files from RedPajamaDataset.
        Preprocesses the data.
        Creates distributed versions of the datasets.
        Returns the DataLoader objects for training data.
    """
    print('Loading Training Data Files...')
    train_data = RedPajamaDataset(tokenizer, 'train')
    # Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
    print("Preprocessing...")
    # Create distributed version of the dataset
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(train_data, batch_size=Ben_batch_size, shuffle=True, collate_fn=collate_fn)
    return TrainingChatData


def load_PileData():
    """
    Parameters
    None

    Returns
    TrainingChatData : DataLoader
        DataLoader object containing the training data for the model.

    Description This function loads and preprocesses the training data files for a chatbot model using
    the Pile dataset. It performs the following steps:
        Loads the training data files from PileDataset.
        Preprocesses the data.
        Creates distributed versions of the datasets.
        Returns the DataLoader objects for training data.
    """
    print('Loading Training Data Files...')
    train_data = PileDataset(tokenizer, 'train')
    # Instantiate preprocessing class object with current tokenizer and specified train dataset JSON file
    print("Preprocessing...")
    # Create distributed version of the dataset
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(train_data, batch_size=Ben_batch_size, shuffle=True, collate_fn=collate_fn)
    return TrainingChatData


def load_StarCoder(coding_language='python'):
    """
    Parameters
    coding_language : str
        specified dataset subset for https://huggingface.co/datasets/bigcode/starcoderdata

    Returns
    TrainingChatData : DataLoader
        DataLoader object containing the training data for the model.

    Description This function loads and preprocesses the training data files for a chatbot model using
    the StarCoder dataset. It performs the following steps:
        Loads the training data files from StarCoderDataset.
        Preprocesses the data.
        Creates distributed versions of the datasets.
        Returns the DataLoader objects for training data.
    """
    print('Loading Training Data Files...')
    train_data = StarCoderDataset(tokenizer, data_dir=coding_language, split='train')
    print("Preprocessing...")
    print("Distributing Data Sets...")
    TrainingChatData = DataLoader(train_data, batch_size=Ben_batch_size, shuffle=True, collate_fn=collate_fn)
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


def Grab_Tokenizer():
    tokenizerGrab = AutoTokenizer.from_pretrained(tokenizer_type)
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
    11. Write the predicted words and their probabilities to a CSV file.

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

    # Prepare data for CSV
    csv_data = []
    for i, token in enumerate(top_5_tokens):
        word = tokenizer.decode([token])
        probability = top_5_token_probs[i]
        csv_data.append([word, probability])

    # Write data to CSV
    with open('Model_Sample_Inferences.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Word", "Probability"])
        writer.writerows(csv_data)


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
            torch.set_default_tensor_type("torch.FloatTensor")
            with open(file_path, mode='a', newline='') as results_file:
                writer = csv.writer(results_file)
                for batch_num, batch in enumerate(
                        tqdm(trainer_data, desc=f'Training Epoch {epochR}, Batch', leave=True)):
                    if batch is None:
                        # Skip the last empty batch (As the multicore Encoder returns NoneType for last index)
                        continue
                    inputs, attention_masks, targets = batch
                    batch_size = len(inputs)
                    optimizer.zero_grad()
                    outputs = model_pass(inputs, attention_mask=attention_masks, labels=targets)
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
    with open(file_path, mode='w', newline='') as results_file:
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
    tokenizer = Grab_Tokenizer()
    # Model Declaration
    model = BertForMaskedLM.from_pretrained(model_name)
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
        # (Entries Longer than specified number will be truncated and Entries Shorter will be Padded)
        # GPT2 has a max length of 1024 tokens
        # According to OpenAI, the conversion rate of character to token is 4:1
        # Cite: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        # Training RunTime
        print("Fine-tuning...")
        train_model(TrainChatData, model, optim, scheduler, epoch)
        print("successful fine-tuning...")
        print("Testing Model Training Results With Validation Prompts...")
        torch.cuda.empty_cache()
        accelerator.free_memory()
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
