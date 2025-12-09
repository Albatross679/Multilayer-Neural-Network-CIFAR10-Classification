# ECE 5460 Final Project: CIFAR-10 Image Classification

A PyTorch-based convolutional neural network (CNN) implementation for classifying images from the CIFAR-10 dataset. This project achieved **81.78% test accuracy**, significantly exceeding the target of 65%.

## 📋 Project Overview

This project implements a multilayer CNN classifier for the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The implementation focuses on:

- **Network Architecture Improvements**: Deeper and wider CNN with increased filter counts
- **Data Augmentation**: Comprehensive augmentation pipeline to improve generalization
- **Hyperparameter Tuning**: Optimized learning rate, batch size, and regularization
- **Performance Optimization**: Achieved 81.78% test accuracy (16.78% above target)

## 🎯 Key Results

- **Overall Test Accuracy**: **81.78%** (Target: 65%)
- **Best Validation Accuracy**: 82.25%
- **All 10 classes exceed 65% accuracy target**
- **Best performing classes**: Car and Truck (91.00% each)
- **Model Parameters**: 686,282 trainable parameters

### Per-Class Performance

| Class | Accuracy | Status |
|-------|----------|--------|
| car | 91.00% | 🥇 Best |
| truck | 91.00% | 🥇 Best |
| ship | 90.30% | ✅ Excellent |
| horse | 86.90% | ✅ Excellent |
| plane | 83.70% | ✅ Very Good |
| dog | 81.40% | ✅ Very Good |
| frog | 79.90% | ✅ Good |
| deer | 76.20% | ✅ Good |
| bird | 71.70% | ✅ Above Target |
| cat | 65.70% | ⚠️ Lowest (but above target) |

## 🏗️ Architecture

### Improved CNN Architecture

```
Input (3×32×32)
  ↓
Conv1: 3→32 filters (5×5, padding=2) + ReLU + MaxPool(2×2)
  ↓
Conv2: 32→64 filters (5×5, padding=2) + ReLU + MaxPool(2×2)
  ↓
Conv3: 64→128 filters (3×3, padding=1) + ReLU + MaxPool(2×2)
  ↓
Flatten: 128×4×4 = 2048
  ↓
FC1: 2048→256 + ReLU + Dropout(0.5)
  ↓
FC2: 256→128 + ReLU
  ↓
FC3: 128→10 (Output)
```

**Key Features:**
- 3 convolutional layers with increasing filter counts (32→64→128)
- 3 fully connected layers with dropout regularization
- Total parameters: 686,282

## 🚀 Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended) or CPU
- pip package manager

### Setup

1. Clone or download this repository:
```bash
cd /path/to/final_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

For CPU-only installation:
```bash
pip install torch torchvision torchaudio
```

## 📖 Usage

### Running the Notebook

1. Open `ece5460FinalProject.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells sequentially
3. The notebook will:
   - Download CIFAR-10 dataset automatically
   - Apply data augmentation to training set
   - Train the improved CNN model
   - Evaluate on test set and display results

### Training Configuration

- **Optimizer**: SGD with momentum (lr=0.01, momentum=0.9, weight_decay=5e-4)
- **Learning Rate Schedule**: StepLR (step_size=10, gamma=0.5)
- **Batch Size**: 32
- **Epochs**: 30
- **Train/Validation Split**: 80%/20%
- **Device**: CUDA (GPU) if available, otherwise CPU

### Data Augmentation

The training pipeline includes:
- **Random Horizontal Flip** (p=0.5)
- **Random Rotation** (±10 degrees)
- **Random Affine Translation** (±10% shift)
- **Color Jitter** (brightness, contrast, saturation, hue variations)

## 📁 Project Structure

```
final_project/
├── ece5460FinalProject.ipynb    # Main project notebook
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── report.md                     # Detailed project report
├── FinalProject_5460.pdf         # Project specification
└── Image_Processing_Final_Project_Report.pdf  # Project report PDF
```

## 🔧 Technical Details

### Dependencies

- **PyTorch** ≥2.0.0 (with CUDA support)
- **torchvision** ≥0.15.0
- **numpy** ≥1.24.0
- **matplotlib** ≥3.7.0
- **seaborn** ≥0.12.0
- **scikit-learn** ≥1.3.0
- **scikit-image** ≥0.21.0
- **opencv-python** ≥4.8.0

### Training Details

- **Training Time**: ~15-20 minutes for 30 epochs (on GPU)
- **Model Checkpoint**: Best model saved based on validation accuracy
- **Final Evaluation**: Model loaded from best checkpoint for test set evaluation

## 📊 Performance Metrics

### Overall Metrics
- **Test Accuracy**: 81.78%
- **Macro Average Precision**: 0.8230
- **Macro Average Recall**: 0.8178
- **Macro Average F1-Score**: 0.8188

### Training Progress
The model showed consistent improvement:
- Epoch 0: 42.83% validation accuracy
- Epoch 10: 74.49% validation accuracy
- Epoch 20: 81.36% validation accuracy
- Epoch 29: 81.84% validation accuracy
- Best: 82.25% validation accuracy

## 🎓 Course Information

- **Course**: ECE 5460 (Image Processing)
- **Term**: Autumn 2025
- **Project**: Multilayer Neural Network CIFAR10 Classification

## 📝 Key Techniques Implemented

1. **Architecture Improvements**
   - Increased filter counts (32→64→128)
   - Additional convolutional layer
   - Deeper fully connected layers
   - Dropout regularization

2. **Data Augmentation**
   - Geometric transformations (flip, rotation, translation)
   - Color space augmentations

3. **Regularization**
   - Dropout (0.5) in fully connected layers
   - Weight decay (L2 regularization)

4. **Training Optimization**
   - Learning rate scheduling
   - Momentum-based SGD
   - Validation-based early model selection

## 🔮 Future Improvements

Potential enhancements for even better performance:
- Batch normalization for faster convergence
- Modern architectures (ResNet, VGG-style blocks)
- Advanced augmentation (CutOut, MixUp)
- Early stopping based on validation performance
- Alternative optimizers (Adam, AdamW)
- Ensemble methods

## 📄 License

This project is part of an academic course assignment.

## 🙏 Acknowledgments

- CIFAR-10 dataset creators
- PyTorch development team
- ECE 5460 course instructors

---

**Note**: This project was developed as part of the ECE 5460 course requirements. The implementation demonstrates practical application of deep learning techniques for image classification tasks.
