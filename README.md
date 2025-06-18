# Dysarthric Speech Classification using Machine Learning Optimization Techniques

## Video Presentaion
https://drive.google.com/file/d/1-mABonVLbJo4BB6WOTn3mUnVNnkpKUPP/view?usp=sharing 

## Project Overview
This project addresses the classification of dysarthric speech patterns using advanced machine learning and neural network optimization techniques. Dysarthria is a speech disorder that affects motor speech control, making automated detection crucial for assistive technologies. The project compares traditional machine learning approaches with optimized neural networks to identify the most effective classification strategy. We utilize the TORGO dataset containing speech samples from both dysarthric speakers and healthy controls to develop robust classification models.

## Dataset
- **Source**: TORGO Database (University of Toronto)
- **Classes**: Binary classification (Dysarthric vs Control/Healthy speech)
- **Sample Size**: 50 files per class (100 total files)
- **Features**: Audio files processed into MFCC, spectral features, and spectrograms
- **Data Split**: 60% training, 20% validation, 20% testing


## How to Run

### Local Environment
1. Clone the repository:
```bash
git clone https://github.com/yourusername/Dysarthric_Speech_Classification.git
cd Dysarthric_Speech_Classification
```
## Create a virtual Environment

```bash
python -m  venv venv 
```

## Installation and Setup

```bash
pip install -r requirements.txt
```

2. Ensure your dataset is organized as:
```
Dysarthric_Speech_Classification/
├── M_Dys/wav_arrayMic_M01S01/  # Dysarthric speech files
├── M_Con/wav_arrayMic_MC01S01/ # Control speech files
└── dysarthria2.ipynb           # Main notebook
```

3. Run the Jupyter notebook:
```bash
jupyter notebook dysarthria2.ipynb
```

### Google Colab
1. Upload the dataset to Google Colab or mount Google Drive
2. Modify the paths in Cell 1:
```python
CONTROL_PATH = "/content/M_Con"
DYSARTHRIC_PATH = "/content/M_Dys"
```
3. Run all cells sequentially

### Execution Order
1. **Cell 1**: Project setup and data loading
2. **Cell 2**: Feature extraction functions
3. **Cell 3**: Data preparation and splitting
4. **Cell 4**: Logistic Regression with hyperparameter tuning
5. **Cell 5**: Neural network data generator
6. **Cell 6**: Simple neural network (baseline)
7. **Cell 7**: Optimized NN Instance 2 (Adam + L2)
8. **Cell 8**: Optimized NN Instance 3 (RMSprop + L1 + BatchNorm)
9. **Cell 9**: Advanced NN Instance 4 (SGD + L1L2)
10. **Cell 10**: Results analysis and comparison
11. **Cell 11**: Visualization
12. **Cell 12**: Final model selection

## Experimental Results

### Model Performance Comparison

| Instance | Optimizer | Regularizer | Epochs | Early Stopping | Layers | Learning Rate | Accuracy | F1-Score | Precision | Recall | Loss |
|----------|-----------|-------------|--------|----------------|--------|---------------|----------|----------|-----------|--------|------|
| Logistic Regression | - | L1/L2 (Grid Search) | - | - | - | - | 0.850 | 0.824 | 1.000 | 0.700 | - |
| Instance 1 (Simple) | Adam (default) | None | 10 | No | 5 | 0.001 | 0.950 | 0.947 | 1.000 | 0.900 | 0.283 |
| Instance 2 (Opt) | Adam | L2 (0.01) | 20 | Yes | 6 | 0.0005 | 0.950 | 0.947 | 1.000 | 0.900 | 0.343 |
| Instance 3 (Opt) | RMSprop | L1 (0.01) + BatchNorm | 15 | Yes | 7 | 0.001 | 0.500 | 0.615 | 0.500 | 0.800 | 2.367 |
| Instance 4 (Advanced) | SGD | L1L2 (0.005, 0.005) | 30 | Yes | 8 | 0.01 | 0.900 | 0.889 | 1.000 | 0.800 | 1.587 |

### Key Findings

#### Best Performing Model
**Instance 1 (Simple) and Instance 2 (Optimized)** both achieved the highest performance with:
- **Accuracy**: 95.0%
- **F1-Score**: 94.7%
- **Precision**: 100.0%
- **Recall**: 90.0%

**Instance 1 (Simple NN)** specifications:
- **Optimizer**: Adam with default settings
- **Regularization**: None
- **Architecture**: Lightweight 5-layer network

**Instance 2 (Optimized)** specifications:
- **Optimizer**: Adam with learning rate 0.0005
- **Regularization**: L2 regularization (0.01)
- **Architecture**: Lightweight 6-layer network with early stopping

#### Optimization Techniques Impact
1. **L2 Regularization (Instance 2)**: Maintained high F1-score of 94.7% while adding regularization
2. **L1 + BatchNormalization (Instance 3)**: Achieved 61.5% F1-score with feature sparsity but lower overall performance
3. **L1L2 + Advanced Architecture (Instance 4)**: Good performance at 88.9% F1-score with combined regularization
4. **Early Stopping**: Prevented overfitting in optimized models (Instances 2, 3, 4)

#### Classical ML vs Neural Networks
- **Logistic Regression**: 82.4% F1-score with optimized hyperparameters
- **Best Neural Network**: 94.7% F1-score (Instance 1 & 2)
- **Performance Gap**: Neural networks outperformed classical ML by 12.3% F1-score

#### Logistic Regression (Best Classical ML)
- **Regularization**: L1 penalty
- **C parameter**: 0.1 (inverse regularization strength)
- **Solver**: liblinear
- **Max iterations**: 1000
- **Cross-validation**: 5-fold for hyperparameter tuning
- **Performance**: 85.0% accuracy, 82.4% F1-score, 100% precision, 70% recall

#### Best Neural Network Configuration
- **Best Performers**: Instance 1 (Simple) and Instance 2 (Optimized) - both achieved 95.0% accuracy
- **Instance 2 Features**:
  - Optimizer: Adam with learning rate 0.0005
  - Regularization: L2 (0.01)
  - Early stopping patience: Applied
  - Architecture: Lightweight convolutional network optimized for GitHub deployment
- **Instance 1 Insight**: Demonstrates that simple architectures can achieve excellent performance

### Most Effective Combination
Both the **Simple Neural Network (Instance 1)** and **Optimized Neural Network (Instance 2)** proved most effective:
- Excellent generalization with 95.0% test accuracy and 94.7% F1-score
- Perfect precision (100%) with good recall (90%)
- Efficient feature learning through lightweight architecture
- Instance 2 adds L2 regularization for additional robustness
- Optimized for deployment with minimal model size

## References
- TORGO Database: University of Toronto
- TensorFlow/Keras framework
- Scikit-learn machine learning library
- Librosa audio processing library
