# Author: Dr. Jian Gong
# Project: University of Wyoming - ARCC GPU Benchmarking Toolkit
# Model: Train yolov8n model on custom labelled dataset (full coco dataset) for object detection benchmarking
# Backend: PyTorch

from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="dataset.yaml", epochs=4, batch=96)  # train the model for 4 epochs
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
