GPT2-Medium Chatbot
This repository contains training code and uses a GPT2-Medium chatbot model on the OpenWebtext dataset. The model uses PyTorch and uses the built-in Adam optimizer for training.

Model Details
Model: GPT2-Large
Dataset: OpenWebtext
Batch Size: 30
Context Length: 128 tokens per entry
Gradient Accumulation Value: 1
Gradient Clipping Value: 1
Loss Function: Cross-Entropy Loss

System Requirements
The model is designed to run on an NVIDIA A30 (24GB Variant) GPU with 22.6 GB utilization. It is recommended to have a compatible GPU and sufficient memory to train and use the model effectively.
