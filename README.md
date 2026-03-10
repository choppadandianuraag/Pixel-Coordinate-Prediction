
## Project Overview
Deep learning model to predict pixel coordinates in 50x50 grayscale images.
A CNN is trained to locate the position of a single white pixel (value=255) on a black background.

## Dataset
- **2,500 unique images** — one for every (x, y) position in the 50×50 grid
- **Leakage-free splits**: All positions are enumerated, shuffled, and partitioned into mutually exclusive sets:
  - Train: 2,000 | Validation: 250 | Test: 250
- **No duplicate images** can appear across splits (each position exists exactly once)
- Labels are encoded in filenames (e.g., `train_00001_x14_y42.png`) to guarantee image-label alignment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choppadandianuraag/Pixel-Coordinate-Prediction/blob/main/Model_Training.ipynb)

## Project Structure
```
├── images/
│   ├── train/                  # 2,000 training images
│   ├── val/                    # 250 validation images
│   └── test/                   # 250 test images
├── models/
│   └── best_pixel_model_v2.h5  # Trained model weights
├── src/
│   └── logs/                   # TensorBoard logs
├── generate_dataset.py         # Leakage-free dataset generation
├── Model_Training.ipynb        # Main training notebook
├── pixel_coordinates.csv       # Ground-truth coordinates & split info
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Dataset Generation Rationale

### Problem Analysis
The task requires predicting (x, y) coordinates of a single white pixel (value 255) in a 50x50 grayscale image.

### Dataset Design Decisions

1. **Image Size: 50x50 pixels**
   - Specified in problem statement
   - Small enough for fast training
   - Large enough to test model's spatial understanding

2. **Dataset Size: 10,000 total images**
   - Train: 8,000 images (80%)
   - Validation: 1,000 images (10%)
   - Test: 1,000 images (10%)
   - Rationale: Sufficient samples to learn spatial patterns without overfitting

3. **Pixel Placement: Uniform Random Distribution**
   - Each pixel position (x, y) has equal probability
   - Ensures model learns all regions of the image equally
   - Prevents bias toward center or edges
   - Real-world scenario: pixel could appear anywhere

4. **Image Format: Grayscale (single channel)**
   - Simplifies problem - only spatial location matters
   - Reduces model complexity
   - Faster training and inference

5. **Single Pixel Per Image**
   - As specified in problem statement
   - Binary classification in spatial domain
   - Tests model's ability to localize precisely

### Why This Approach Works
- **Regression problem**: Predicting continuous (x, y) coordinates
- **Spatial learning**: CNN learns to detect and localize the bright pixel
- **Balanced dataset**: No class imbalance across image regions

## Installation

### Prerequisites
- Python 3.11+
- macOS with Apple Silicon (tensorflow-macos) or CUDA GPU

### Setup
```bash
# Clone the repository
git clone https://github.com/choppadandianuraag/Pixel-Coordinate-Prediction/
cd Pixel-Coordinate-Prediction
# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Generate Dataset
```bash
python generate_dataset.py
```
This enumerates all 2,500 positions, shuffles, splits, and saves images with coordinate-encoded filenames.

### 2. Train & Evaluate
Open `Model_Training.ipynb` and run all cells sequentially to:
1. Load images with labels parsed from filenames (no CSV dependency for alignment)
2. Verify zero overlap between splits (leakage assertion)
3. Train the CNN model
4. Evaluate on the held-out test set
5. Generate visualizations

## Model Architecture
**Simple CNN v2** - Best performing model:
- Conv2D(16) → BatchNorm → ReLU → MaxPool
- Conv2D(32) → BatchNorm → ReLU → MaxPool  
- Conv2D(64) → BatchNorm → ReLU
- Flatten → Dense(64) → Dense(2)

### Training Configuration
- **Loss**: Huber Loss
- **Optimizer**: Adam (lr=0.001)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Epochs**: 30 (with early stopping)

## Results
Retrain the model using the notebook to get fresh results on the leakage-free dataset.

### Visualizations
See `Model_Training.ipynb` for:
- Predicted vs actual scatter plots
- Error distribution histograms
- Sample predictions overlaid on images
- Error vs position analysis

## Dependencies
See `requirements.txt`:
- TensorFlow 2.15.0 (tensorflow-macos on Apple Silicon)
- NumPy, Pandas, Matplotlib, OpenCV
