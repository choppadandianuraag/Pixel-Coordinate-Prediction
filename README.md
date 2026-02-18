
## Project Overview
Deep learning model to predict pixel coordinates in 50x50 grayscale images with sub-pixel accuracy.

## Performance
- **Test Mean Error**: 0.26 pixels
- **Test Median Error**: 0.25 pixels  
- **Accuracy**: 99.5% positional accuracy

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choppadandianuraag/Pixel-Coordinate-Prediction/blob/main/Model_Training.ipynb)

## Project Structure
```
├── images/
│   ├── train/                  # Training images
│   ├── val/                    # Validation images
│   └── test/                   # Test images
├── models/
│   └── best_pixel_model_v2.h5  # Trained model weights
├── notebooks/
│   └── Model_Training.ipynb    # Main training notebook
├── src/
│   └── utils.py                # Helper scripts (if any)
├── generate_dataset.py         # Dataset generation script
├── main.py                     # Main execution script
├── pixel_coordinates.csv       # Ground-truth pixel coordinates
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
- Python 3.9+
- CUDA-compatible GPU (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/choppadandianuraag/Pixel-Coordinate-Prediction/
cd pixel-prediction

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Using Jupyter Notebook (Recommended)
```bash
jupyter notebook notebooks/Model_Training.ipynb
```
Run all cells sequentially to:
1. Load and preprocess data
2. Train the CNN model  
3. Evaluate on test set
4. Generate visualizations


### Generate New Dataset (Optional)
```bash
python generate_dataset.py
```

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
- **Best val_loss**: 0.031

## Results

### Quantitative Results
| Metric | Value |
|--------|-------|
| Test MAE | 0.26 pixels |
| Test Median Error | 0.25 pixels |
| Max Error | <0.5 pixels |

### Visualizations
See `notebooks/Model_Training.ipynb` for:
- Training/validation loss curves
- Predicted vs actual scatter plots
- Error distribution histograms
- Sample predictions overlaid on images

## Code Quality
- ✅ PEP8 compliant
- ✅ Comprehensive comments
- ✅ Modular design
- ✅ Clear documentation

## Dependencies
See `requirements.txt` for full list:
- TensorFlow 2.15.0
- NumPy, Pandas, Matplotlib
- Scikit-learn
