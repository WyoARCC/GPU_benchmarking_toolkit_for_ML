# Author: Tyson Limato
# project: GPU Benchmarking
# Model: Training GPT2 with Wikitext103-v1 from HuggingFace
# Backend: Pytorch
# High level Imports
import os
import csv
import time
# Pytorch Imports
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.cuda
# Hugging Face Imports
from transformers import GPT2LMHeadModel, AutoTokenizer
from DataSetsForLLM import WikiTextDataset, OpenWebTextDataset
from accelerate import Accelerator
# Progress Bar Imports
from tqdm import tqdm

# Define Device for Training
num_gpus = os.environ.get('CUDA_VISIBLE_DEVICES') if not None else 1
accelerator = Accelerator(mixed_precision='fp16')
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

    Description
    This function loads and preprocesses the training and validation data files for a chatbot model using the WikiText dataset. It performs the following steps:
        Loads the training data files.
        Loads the validation data files.
        Preprocesses the data.
        Creates distributed versions of the datasets.
        Returns the DataLoader objects for training and validation data.
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
    TrainingChatData = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collate_fn)
    ValidatingChatData = DataLoader(validation_data, batch_size=8, collate_fn=collate_fn)
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
    train_data = OpenWebTextDataset(tokenizer)
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
    tokenizerGrab = AutoTokenizer.from_pretrained("gpt2-large")
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

def infer(prompt):
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
    """
    Parameters
    ----------
    trainer_data : DataLoader
        DataLoader object containing the training data for the model.
    val_data : DataLoader
        DataLoader object containing the validation data for the model.
    model_pass : transformer.GPT2LMHeadModel
        The model to be trained and validated.
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
    This function performs the training and validation process for a given model using the
    specified training and validation data. It follows the following steps:

    1. Sets the model to training mode.
    2. Initializes variables for tracking validation loss, number of iterations, and batch size.
    3. Sets the default tensor type to "torch.FloatTensor".
    4. Iterates through each epoch.
    5. Within each epoch, iterates through each batch of the training data.
    6. If the batch is None (empty), skips to the next iteration which accounts for the last batch being empty.
    7. Retrieve inputs and targets from the batch.
    8. Zeroes the gradients of the optimizer.
    9. Passes the inputs through the model to get the outputs.
    10. Calculates the loss using the specified criterion.
    11. Performs backward propagation of gradients using the accelerator (if available).
    12. Updates the model parameters based on the gradients and optimizer's update rule.
    13. Adjusts the learning rate using the scheduler.
    14. Performs gradient accumulation if the number of iterations reaches the specified accumulation steps.
    15. Frees memory using the accelerator.
    16. Calculates the epoch time and throughput.
    17. Sets the model to evaluation mode.
    18. Iterates through each batch of the validation data.
    19. If the batch is None (empty), skips to the next iteration.
    20. Retrieve inputs and targets from the batch.
    21. Passes the inputs through the model to get the outputs.
    22. Calculates the validation loss using the specified criterion.
    23. Gathers the validation loss using the accelerator.
    24. Frees CUDA memory and accelerator memory.
    25. Divides the accumulated validation loss by the number of validation batches.
    26. Prints the test response for the current epoch using the infer function.
    27. Saves the model's state dictionary to a file.
    28. Writes the validation loss, epoch time, and throughput to a CSV file.
    29. Waits for all processes to synchronize using the accelerator.
    """
    # Set the model to training mode
    model_pass.train()
    # Gradient Clipping Variable
    max_grad_norm = 1
    val_loss = 0
    num_iterations = 0
    # Gradient Accumulation Variable
    accumulation_steps = 1
    # Specific to the GPT2 Model, Requires Tensors in Float Form
    torch.set_default_tensor_type("torch.FloatTensor")
    for epochR in range(epoch_pass):
        # Training loop
        start = time.time()
        batch_size = 0
        for batch in tqdm(trainer_data, desc=f'Training Epoch {epochR}, Batch', leave=True):
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
            # Gradient Accumulation
            if num_iterations == accumulation_steps:
                # Gradient clipping
                clip_grad_norm_(model_pass.parameters(), max_grad_norm)
                optimizer.step()
                scheduler_pass.step(loss)
                num_iterations = 0
        # Free Up any unnecessary allocations of VRAM
        accelerator.free_memory()
        end = time.time()
        # Time For the Epoch to complete (Verifies tqdm estimation)
        epochTime = end - start
        # Calculate Throughput of the System
        throughput = batch_size * len(trainer_data) / epochTime

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
                writer.writerow(['Epoch', 'Validation Loss', 'Time', 'Throughput'])
            writer.writerow([epochR + 1, val_loss, epochTime, throughput])

        accelerator.wait_for_everyone()


if __name__ == '__main__':
    # tokenizer Declaration and special token Declaration
    tokenizer = GPT2_Tokenizer()
    # Model Declaration
    model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    model.resize_token_embeddings(len(tokenizer))
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
        # (Entries Longer than specified number will be truncated and Entries Shorter will be Padded with empty tokens)
        # GPT2 has a max length of 1024 tokens
        # According to OpenAI, the conversion rate of character to token is 4:1
        # Cite: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        "Currently this global variable is not in use!"
        os.environ['max_tok_length'] = str(256)
        # Training RunTime
        print("Fine-tuning...")
        train_and_validate(TrainChatData, ValidationChatData, model, optim, scheduler, epoch)
        print("successful fine-tuning...")
        print("Testing Model Training Results With Validation Prompt...")
        torch.cuda.empty_cache()
        accelerator.free_memory()

        for x in range(10):
            ModelGeneration = infer(
                "Albert Einstein was ")
            print(ModelGeneration)

    except KeyboardInterrupt:
        print("Aborted by the User")
