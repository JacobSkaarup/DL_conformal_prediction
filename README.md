# Deep Learning Conformal Prediction

A PyTorch-based implementation of conformal prediction methods for uncertainty quantification with deep neural networks, specifically applied to image classification tasks with CIFAR-10.

## Overview

Conformal prediction is a framework for creating prediction sets with guaranteed coverage properties. This project implements conformal prediction techniques (TPS, APS, RAPS, and DAPS) on top of deep learning models to provide uncertainty estimates alongside classifications. Unlike standard confidence scores, conformal predictions offer formal statistical guarantees on the fraction of correct predictions.

## Features

- **Conformal Prediction Library**: Core implementation of conformal prediction methods
- **CIFAR-10 Training**: Pre-built pipeline for training ResNet-18 on CIFAR-10
- **Uncertainty Quantification**: Obtain prediction sets with coverage guarantees
- **PyTorch Integration**: Seamless integration with PyTorch models using scikit-learn interfaces
- **Jupyter Notebooks**: Interactive examples for experimentation and analysis

## Project Structure

```
├── conformal/            # Core conformal prediction library
│   ├── cp_lib.py          # Main conformal prediction implementations
│   ├── smoothers.py       # Smoothing functions for conformal methods
│   └── __init__.py
├── src/                 # Data and model training utilities
│   ├── data.py           # Data loading and preprocessing
│   ├── train.py          # Training and evaluation functions
│   └── __pycache__/
├── dataset/             # CIFAR-10 dataset
│   └── cifar-10-batches-py/
├── results/             # Trained model checkpoints and predictions
├── project_main.ipynb    # Main project workflow notebook
├── trainCifar10.ipynb    # CIFAR-10 training notebook
├── trainCifar10.py       # CIFAR-10 training script
├── setup.py              # Package configuration
├── requirements.txt      # Required packages
└── README.md

```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- NumPy, scikit-learn, tqdm

### Setup

1. Clone the repository:
```bash
git clone https://github.com/JacobSkaarup/DL_conformal_prediction.git
cd DL_conformal_prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install the package:
```bash
pip install -e .
```

## Quick Start

### Training a Model

Use the provided training script to train a ResNet-18 on CIFAR-10:

```bash
python trainCifar10.py
```

Or run the interactive 'trainCifar10.ipynb' notebook made for Google Colab:


### Running Conformal Prediction

See `project_main.ipynb` for examples of:
- Loading trained models
- Calibrating conformal prediction methods
- Generating prediction sets with coverage guarantees
- Evaluating coverage and set size metrics

## Usage Examples

### Basic Conformal Prediction

```python
from conformal.cp_lib import TorchAdapter
import torch

# Wrap your PyTorch model
model = torch.load('results/cifar10_resnet18.pth')
adapter = TorchAdapter(model, classes=range(10), device='cuda')

# Get prediction probabilities
X_test = # ... your test data ...
proba = adapter.predict_proba(X_test)

# Apply conformal prediction with desired significance level
# (Examples available in project_main.ipynb)
```

## Key Components

### `conformal/cp_lib.py`
- **TorchAdapter**: Wrapper to make PyTorch models compatible with scikit-learn interface
    - Conformal prediction methods and utilities for uncertainty quantification

### `conformal/smoothers.py`
- Smoothing techniques for improved conformal prediction performance

### `src/train.py`
- Model training routines
- Evaluation functions with prediction caching
- Hook-based feature extraction

### `src/data.py`
- CIFAR-10 data loading and preprocessing
- Train/validation/test split handling

## Results

Trained models and predictions are saved in `results/`:
- `cifar10_resnet18.pth` - Trained ResNet-18 model
- `cal_predictions.pth` - Calibration set predictions
- `val_predictions.pth` - Validation set predictions
- `test_predictions.pth` - Test set predictions

## Notebooks

1. **trainCifar10.ipynb** - Complete pipeline for training and validating the model
2. **project_main.ipynb** - Main analysis and experimentation workflow

## Dependencies

- **torch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **tqdm**: Progress bars

See `setup.py` for complete dependency list.

## Contributing

Contributions are welcome! Please ensure code follows the existing style and includes docstrings.

## License

This project is open source and available under the MIT License.

## References

For more on the implemented conformal predictions, see:
- [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.03025)
- [Conformal Prediction Sets for Graph Neural Networks](https://proceedings.mlr.press/v202/h-zargarbashi23a.html)

## Contact

For questions or issues, please open an issue on GitHub or contact the repository maintainer.
