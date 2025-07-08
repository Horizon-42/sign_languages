import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import os
from tqdm import tqdm # For a progress bar
from dataset import *
from model import HandGestureCNN
# --- 1. Configuration ---
# Path to your best saved model checkpoint
MODEL_PATH = 'runs/train/best.pt'
INFERENCE_DATA_PATH = 'data/thws-mai-idl-ss-25-sign-language/SignLanguage_kaggle/todo.pth'
INFERENCE_EXAMPLE_PATH = "data/thws-mai-idl-ss-25-sign-language/SignLanguage_kaggle/todo_example.pth"
# Output CSV file path
OUTPUT_CSV_PATH = './inference_results.csv'

# Number of classes your model was trained on
NUM_CLASSES = 24

# Batch size for inference (can be larger than training batch size if memory allows)
INFERENCE_BATCH_SIZE = 64

# Device to use (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device for inference: {device}")

# load the model
model = HandGestureCNN(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

# --- 2. Test inference accuracy through todo_example ---
print(f"Loading inference example data from {INFERENCE_EXAMPLE_PATH}...")
example_data = read_tensor_dataset(
    INFERENCE_EXAMPLE_PATH)
example_dataset = SignLanguageDataset(
    example_data, transform=v2.Resize(size=(64, 64)))
example_dataloder = DataLoader(
    example_dataset, batch_size=INFERENCE_BATCH_SIZE, shuffle=False)
example_data_size = len(example_dataset)

print("Starting inference test...")
example_corrects = 0
with torch.no_grad():  # 不计算梯度
    for inputs, labels in example_dataloder:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        example_corrects += torch.sum(preds == labels.data)

    test_acc = example_corrects.double() / example_data_size
    print(f'Test Accuracy: {test_acc:.4f}')

# --- 3. Perform Inference ---
print(f"Loading inference data from {INFERENCE_DATA_PATH}...")
# Create a TensorDataset for inference
inference_dataset = SignLanguageDataset(read_tensor_dataset(
    INFERENCE_DATA_PATH), transform=v2.Resize(size=(64, 64)))
inference_dataloader = DataLoader(
    inference_dataset, batch_size=64, shuffle=False)

print("Starting inference...")
predictions = []
image_indices = range(len(inference_dataset))
print(len(inference_dataset))

with torch.no_grad(): # Disable gradient calculation for inference (saves memory, faster)
    for inputs, _ in tqdm(inference_dataloader):
        inputs = inputs.to(device)

        # Apply normalization if not already done in data loading/transform
        # If your 'all_images' are already 0-1 float, apply standard normalization here
        # normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # inputs = normalize_transform(inputs)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1) # Get the predicted class index

        # Store results
        predictions.extend(preds.cpu().tolist()) # Move to CPU and convert to Python list

print("Inference complete.")

predictions = [idx_to_label(lb) for lb in predictions]

print(len(predictions))

# --- 5. Save Results to CSV ---
results_df = pd.DataFrame({
    'ID': image_indices,
    'Label': predictions
})

results_df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"Inference results saved to {OUTPUT_CSV_PATH}")

print("\nDone!")