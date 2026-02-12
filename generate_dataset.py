"""
Generate Synthetic Dataset for Pixel Coordinate Prediction
Creates 50x50 grayscale images with a single white pixel (value 255)
All other pixels are 0 (black). The task is to predict the (x, y) coordinates.

Dataset Rationale:
1. Image size: 50x50 pixels - Large enough to be challenging but computationally feasible
2. Single pixel: Makes the problem well-defined and avoids ambiguity
3. Grayscale: Simplifies the problem (single channel) while maintaining complexity
4. Uniform distribution: Ensures model learns all coordinate positions equally
5. Large dataset: 10,000+ samples for robust deep learning training
"""

import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
IMAGE_SIZE = 50  # 50x50 pixels
N_TRAIN = 8000   # Training samples
N_VAL = 1000     # Validation samples
N_TEST = 1000    # Test samples
N_TOTAL = N_TRAIN + N_VAL + N_TEST

print("="*70)
print(" PIXEL COORDINATE DATASET GENERATOR ".center(70, "="))
print("="*70)
print(f"\nConfiguration:")
print(f"  - Image size: {IMAGE_SIZE}x{IMAGE_SIZE} pixels")
print(f"  - Total samples: {N_TOTAL:,}")
print(f"  - Train: {N_TRAIN:,} | Val: {N_VAL:,} | Test: {N_TEST:,}")
print(f"  - Pixel value: 255 (white) on black background")

# Create directories
os.makedirs('data/raw/images/train', exist_ok=True)
os.makedirs('data/raw/images/val', exist_ok=True)
os.makedirs('data/raw/images/test', exist_ok=True)

def generate_image_with_pixel(x, y, size=IMAGE_SIZE):
    """
    Generate a grayscale image with a single white pixel at (x, y)
    
    Args:
        x: x-coordinate (0 to size-1)
        y: y-coordinate (0 to size-1)
        size: Image dimensions
    
    Returns:
        numpy array of shape (size, size)
    """
    image = np.zeros((size, size), dtype=np.uint8)
    image[y, x] = 255  # Note: numpy uses [row, col] = [y, x]
    return image

def generate_dataset_split(n_samples, split_name):
    """Generate dataset for a specific split (train/val/test)"""
    
    data = []
    
    print(f"\nGenerating {split_name} set...")
    for i in tqdm(range(n_samples)):
        # Randomly select coordinates
        x = np.random.randint(0, IMAGE_SIZE)
        y = np.random.randint(0, IMAGE_SIZE)
        
        # Generate image
        image = generate_image_with_pixel(x, y)
        
        # Save image
        filename = f"{split_name}_{i:05d}.png"
        filepath = f"data/raw/images/{split_name}/{filename}"
        Image.fromarray(image, mode='L').save(filepath)
        
        # Store metadata
        data.append({
            'filename': filename,
            'filepath': filepath,
            'x': x,
            'y': y,
            'split': split_name
        })
    
    return data

# Generate all splits
all_data = []
all_data.extend(generate_dataset_split(N_TRAIN, 'train'))
all_data.extend(generate_dataset_split(N_VAL, 'val'))
all_data.extend(generate_dataset_split(N_TEST, 'test'))

# Create DataFrame
df = pd.DataFrame(all_data)

# Save metadata
metadata_path = 'data/raw/pixel_coordinates.csv'
df.to_csv(metadata_path, index=False)

print(f"\n✓ Dataset generated successfully!")
print(f"✓ Images saved to: data/raw/images/")
print(f"✓ Metadata saved to: {metadata_path}")

# Statistics
print(f"\n{'='*70}")
print(" DATASET STATISTICS ".center(70, "="))
print(f"{'='*70}")
print(f"\nDataset Shape: {df.shape}")
print(f"\nSplit Distribution:")
print(df['split'].value_counts())

print(f"\nCoordinate Statistics:")
print(df[['x', 'y']].describe())

print(f"\nCoordinate Range:")
print(f"  x: [{df['x'].min()}, {df['x'].max()}]")
print(f"  y: [{df['y'].min()}, {df['y'].max()}]")

# Visualize distribution
print(f"\nCoordinate Distribution (should be uniform):")
x_history, x_bins = np.historyogram(df['x'], bins=10)
y_history, y_bins = np.historyogram(df['y'], bins=10)
print(f"\nx-coordinate distribution across 10 bins:")
for i, count in enumerate(x_history):
    bar = '█' * int(count / x_history.max() * 40)
    print(f"  [{x_bins[i]:4.1f}-{x_bins[i+1]:4.1f}]: {bar} ({count})")

# Show example
print(f"\n{'='*70}")
print(" EXAMPLE SAMPLES ".center(70, "="))
print(f"{'='*70}")
print(df.head(10))

print(f"\n{'='*70}")
print(" DATASET READY FOR TRAINING! ".center(70, "="))
print(f"{'='*70}")
print("\nRationale:")
print("✓ Uniform distribution ensures no coordinate bias")
print("✓ Large dataset (10K samples) enables robust CNN training")
print("✓ Train/Val/Test split allows proper model evaluation")
print("✓ Single pixel makes the regression target unambiguous")
print("✓ 50x50 size balances complexity and computational efficiency")
