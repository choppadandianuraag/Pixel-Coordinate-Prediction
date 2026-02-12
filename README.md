
## Project Overview
Deep learning model to predict pixel coordinates in 50x50 grayscale images with sub-pixel accuracy.

## Performance
- **Test Mean Error**: 0.26 pixels
- **Test Median Error**: 0.25 pixels  
- **Accuracy**: 99.5% positional accuracy

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

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup
```bash
# Clone the repository
git clone <your-repo-url>
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
