"""
Generate Synthetic Dataset for Pixel Coordinate Prediction
Creates 50x50 grayscale images with a single white pixel (value 255)
All other pixels are 0 (black). The task is to predict the (x, y) coordinates.

IMPORTANT: To avoid data leakage, we enumerate ALL unique (x, y) positions
in the 50x50 grid (2,500 total), shuffle them, and split into train/val/test.
This guarantees zero overlap between splits — no test image can appear in training.

Dataset Rationale:
1. Image size: 50x50 pixels - Large enough to be challenging but computationally feasible
2. Single pixel: Makes the problem well-defined and avoids ambiguity
3. Grayscale: Simplifies the problem (single channel) while maintaining complexity
4. Unique positions per split: Prevents data leakage between train/val/test
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
TOTAL_POSITIONS = IMAGE_SIZE * IMAGE_SIZE  # 2,500 unique positions
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

N_TRAIN = int(TOTAL_POSITIONS * TRAIN_RATIO)  # 2,000
N_VAL = int(TOTAL_POSITIONS * VAL_RATIO)       # 250
N_TEST = TOTAL_POSITIONS - N_TRAIN - N_VAL     # 250

print("="*70)
print(" PIXEL COORDINATE DATASET GENERATOR ".center(70, "="))
print("="*70)
print(f"\nConfiguration:")
print(f"  - Image size: {IMAGE_SIZE}x{IMAGE_SIZE} pixels")
print(f"  - Total unique positions: {TOTAL_POSITIONS:,}")
print(f"  - Train: {N_TRAIN:,} | Val: {N_VAL:,} | Test: {N_TEST:,}")
print(f"  - Pixel value: 255 (white) on black background")
print(f"  - NO overlap between splits (leakage-free)")

# Create directories
os.makedirs('images/train', exist_ok=True)
os.makedirs('images/val', exist_ok=True)
os.makedirs('images/test', exist_ok=True)


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


# Generate ALL unique (x, y) positions and shuffle
all_positions = [(x, y) for x in range(IMAGE_SIZE) for y in range(IMAGE_SIZE)]
np.random.shuffle(all_positions)

# Split into train/val/test with NO overlap
train_positions = all_positions[:N_TRAIN]
val_positions = all_positions[N_TRAIN:N_TRAIN + N_VAL]
test_positions = all_positions[N_TRAIN + N_VAL:]

assert len(set(map(tuple, train_positions)) & set(map(tuple, val_positions))) == 0, "Train/Val overlap!"
assert len(set(map(tuple, train_positions)) & set(map(tuple, test_positions))) == 0, "Train/Test overlap!"
assert len(set(map(tuple, val_positions)) & set(map(tuple, test_positions))) == 0, "Val/Test overlap!"

print(f"\n✓ Verified: zero overlap between all splits")


def save_split(positions, split_name):
    """Generate and save images for a split. Encodes coordinates in the filename."""
    data = []
    print(f"\nGenerating {split_name} set ({len(positions)} images)...")
    for i, (x, y) in enumerate(tqdm(positions)):
        image = generate_image_with_pixel(x, y)

        # Encode coordinates in filename to allow label extraction without CSV
        filename = f"{split_name}_{i:05d}_x{x}_y{y}.png"
        filepath = os.path.join('images', split_name, filename)
        Image.fromarray(image, mode='L').save(filepath)

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
all_data.extend(save_split(train_positions, 'train'))
all_data.extend(save_split(val_positions, 'val'))
all_data.extend(save_split(test_positions, 'test'))

# Create DataFrame and save
df = pd.DataFrame(all_data)
metadata_path = 'pixel_coordinates.csv'
df.to_csv(metadata_path, index=False)

print(f"\n✓ Dataset generated successfully!")
print(f"✓ Images saved to: images/")
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

# Verify no leakage
train_set = set(map(tuple, df[df['split'] == 'train'][['x', 'y']].values))
val_set = set(map(tuple, df[df['split'] == 'val'][['x', 'y']].values))
test_set = set(map(tuple, df[df['split'] == 'test'][['x', 'y']].values))
print(f"\nLeakage Check:")
print(f"  Train ∩ Val overlap:  {len(train_set & val_set)} positions")
print(f"  Train ∩ Test overlap: {len(train_set & test_set)} positions")
print(f"  Val ∩ Test overlap:   {len(val_set & test_set)} positions")

# Show example
print(f"\n{'='*70}")
print(" EXAMPLE SAMPLES ".center(70, "="))
print(f"{'='*70}")
print(df.head(10))

print(f"\n{'='*70}")
print(" DATASET READY FOR TRAINING! ".center(70, "="))
print(f"{'='*70}")
print("\nRationale:")
print("✓ All 2,500 unique positions used — complete coverage")
print("✓ Zero overlap between train/val/test — no data leakage")
print("✓ Coordinates encoded in filenames — labels always match images")
print("✓ 50x50 size balances complexity and computational efficiency")
