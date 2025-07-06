import torch
from torch.utils.data import random_split
from torchvision.transforms import ToPILImage
import os
import shutil
from PIL import Image
import numpy as np # For robust PIL conversion

# --- 1. Configuration & Data Loading ---
# Please ensure this path points to your .pth file
PTH_FILE_PATH = './data/thws-mai-idl-ss-25-sign-language/SignLanguage_kaggle/old_annotated.pth'
OUTPUT_DIR = './hand_gestures_dataset' # Output directory for structured dataset

# Dataset split ratios
TRAIN_SPLIT_RATIO = 0.7
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15

print(f"Loading data from '{PTH_FILE_PATH}' (assuming it's a list of (image, label) pairs)...")

# TensorDataset directly wraps the image and label Tensors
full_dataset = torch.load(PTH_FILE_PATH, weights_only=False)
print(f"Created a full TensorDataset with {len(full_dataset)} samples.")

total_size = len(full_dataset)
train_size = int(TRAIN_SPLIT_RATIO * total_size)
val_size = int(VAL_SPLIT_RATIO * total_size)
test_size = total_size - train_size - val_size

# Ensure sum of split sizes equals total_size due to integer casting
lengths = [train_size, val_size, test_size]
if sum(lengths) != total_size:
    lengths = [train_size, val_size, total_size - train_size - val_size]
    print(f"Adjusted split sizes to ensure sum matches total: {lengths}")

# Use random_split to divide the dataset into train, validation, and test subsets
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, lengths,
    generator=torch.Generator().manual_seed(42) # For reproducibility
)

print(f"Dataset split: Training set={len(train_dataset)}, Validation set={len(val_dataset)}, Test set={len(test_dataset)}")

# 3. Save Images to File System

def save_dataset_to_folders(dataset_subset, base_dir, phase_name):
    """Saves images from a dataset subset to a specified directory structure."""
    phase_dir = os.path.join(base_dir, phase_name)
    os.makedirs(phase_dir, exist_ok=True)

    # Clear previous data in this phase directory for a clean run
    if os.path.exists(phase_dir) and len(os.listdir(phase_dir)) > 0:
        print(f"Clearing existing contents of '{phase_dir}'...")
        for item in os.listdir(phase_dir):
            item_path = os.path.join(phase_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

    print(f"Saving {phase_name} dataset to '{phase_dir}'...")
    to_pil = ToPILImage()
    class_counts = {}

    for i, (img_tensor, label_tensor) in enumerate(dataset_subset):
        label = label_tensor.item() # Get Python int from tensor
        class_name = f"class_{label}" # E.g., class_0, class_1, ...

        class_dir = os.path.join(phase_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Ensure image Tensor is suitable for ToPILImage (C, H, W or H, W)
        if img_tensor.dim() == 4: # If it unexpectedly has a batch dimension (e.g., from a custom __getitem__ that yields a batch)
            img_tensor = img_tensor.squeeze(0)

        try:
            pil_image = to_pil(img_tensor)
        except Exception as e:
            # Fallback to NumPy conversion if ToPILImage fails
            print(f"Warning: Sample {i} ({phase_name}, label {label}) failed ToPILImage conversion: {e}. Attempting via NumPy.")
            np_image = img_tensor.cpu().numpy()
            if np_image.ndim == 3 and np_image.shape[0] in [1, 3]: # If CHW, convert to HWC
                np_image = np.transpose(np_image, (1, 2, 0))
            if np_image.dtype == np.float32 or np_image.dtype == np.float64:
                np_image = (np_image * 255).astype(np.uint8) # Scale to 0-255 and convert to uint8
            if np_image.ndim == 3 and np_image.shape[2] == 1: # Grayscale (H,W,1) to (H,W)
                np_image = np_image.squeeze(axis=2)
            try:
                pil_image = Image.fromarray(np_image)
            except Exception as inner_e:
                print(f"Critical: Sample {i} failed even NumPy conversion to PIL image: {inner_e}")
                continue # Skip this image

        image_path = os.path.join(class_dir, f"{i:05d}.png") # Save as PNG
        try:
            pil_image.save(image_path)
            class_counts[label] = class_counts.get(label, 0) + 1
        except Exception as save_e:
            print(f"Error saving image '{image_path}': {save_e}")
            continue

    print(f"Finished saving {phase_name} dataset. Saved {sum(class_counts.values())} images. Class counts: {class_counts}")


# Clean up and create new output directory
if os.path.exists(OUTPUT_DIR):
    print(f"Cleaning up existing output directory: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save each subset to folders
save_dataset_to_folders(train_dataset, OUTPUT_DIR, 'train')
save_dataset_to_folders(val_dataset, OUTPUT_DIR, 'val')
save_dataset_to_folders(test_dataset, OUTPUT_DIR, 'test')

print("\nDataset conversion complete! Your dataset is now structured for ImageFolder.")
print(f"You can find the structured dataset in: {os.path.abspath(OUTPUT_DIR)}")
print("Remember to update the `DATA_DIR` in your training script to point to this new directory.")