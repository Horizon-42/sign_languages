import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import os
from tqdm import tqdm # For a progress bar

# --- 1. Configuration ---
# Path to your best saved model checkpoint
MODEL_PATH = './best_resnet101_model.pth' # <--- IMPORTANT: Update this path!
# Path to your inference dataset (the .pth file with images to predict)
INFERENCE_DATA_PATH = './data/thws-mai-idl-ss-25-sign-language/SignLanguage_kaggle/todo.pth' # <--- IMPORTANT: Update this path!
# Output CSV file path
OUTPUT_CSV_PATH = './inference_results.csv'

# Number of classes your model was trained on
NUM_CLASSES = 24

# Batch size for inference (can be larger than training batch size if memory allows)
INFERENCE_BATCH_SIZE = 64

# Device to use (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device for inference: {device}")

# --- 2. Load the Model ---
print(f"Loading model from {MODEL_PATH}...")
# Initialize the model architecture (ResNet-50 as used in training)
model = models.resnet101(pretrained=False) # No need for pretrained weights if loading a full checkpoint
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# Load the saved state_dict
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode (disables dropout, BatchNorm updates)
    print("Model loaded successfully and set to evaluation mode.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please check the path.")
    exit()
except Exception as e:
    print(f"Error loading model state_dict: {e}")
    exit()

# --- 3. Load the Inference Dataset ---
print(f"Loading inference data from {INFERENCE_DATA_PATH}...")
# Create a TensorDataset for inference
inference_dataset = torch.load(INFERENCE_DATA_PATH)
inference_dataloader = DataLoader(inference_dataset, batch_size=INFERENCE_BATCH_SIZE, shuffle=False, num_workers=4)

# --- 4. Perform Inference ---
print("Starting inference...")
predictions = []
image_indices = []

with torch.no_grad(): # Disable gradient calculation for inference (saves memory, faster)
    for i, (inputs, _) in tqdm(enumerate(inference_dataloader), total=len(inference_dataloader), desc="Predicting"):
        inputs = inputs.to(device)

        # Apply normalization if not already done in data loading/transform
        # If your 'all_images' are already 0-1 float, apply standard normalization here
        # normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # inputs = normalize_transform(inputs)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1) # Get the predicted class index

        # Store results
        current_batch_indices = range(i * INFERENCE_BATCH_SIZE, (i * INFERENCE_BATCH_SIZE) + len(inputs))
        image_indices.extend(current_batch_indices)
        predictions.extend(preds.cpu().tolist()) # Move to CPU and convert to Python list

print("Inference complete.")

# --- 5. Save Results to CSV ---
results_df = pd.DataFrame({
    'ID': image_indices,
    'Label': predictions
})

results_df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"Inference results saved to {OUTPUT_CSV_PATH}")

print("\nDone!")