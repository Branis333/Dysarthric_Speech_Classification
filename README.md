# Dysarthric Speech Classification using Machine Learning Optimization Techniques

## Project Overview
This project addresses the classification of dysarthric speech patterns using advanced machine learning and neural network optimization techniques. Dysarthria is a speech disorder that affects motor speech control, making automated detection crucial for assistive technologies. The project compares traditional machine learning approaches with optimized neural networks to identify the most effective classification strategy. We utilize the TORGO dataset containing speech samples from both dysarthric speakers and healthy controls to develop robust classification models.

## Dataset
- **Source**: TORGO Database (University of Toronto)
- **Classes**: Binary classification (Dysarthric vs Control/Healthy speech)
- **Sample Size**: 50 files per class (100 total files)
- **Features**: Audio files processed into MFCC, spectral features, and spectrograms
- **Data Split**: 60% training, 20% validation, 20% testing

## Installation and Setup

### Prerequisites
```bash
pip install tensorflow>=2.15.0
pip install librosa
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install joblib
```

### For Google Colab
```python
!pip install librosa
```

## How to Run

### Local Environment
1. Clone the repository:
```bash
git clone https://github.com/yourusername/Dysarthric_Speech_Classification.git
cd Dysarthric_Speech_Classification
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

### Hyperparameter Analysis

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

## Summary of Findings

### Exceptional Performance Achieved
This project achieved remarkably high performance on dysarthric speech classification:
- **Best Models**: Instance 1 (Simple) and Instance 2 (Optimized) both achieved 95.0% accuracy
- **F1-Score**: 94.7% demonstrating excellent balance of precision and recall
- **Perfect Precision**: 100% precision achieved by neural networks and logistic regression
- **Strong Recall**: 90% recall for neural networks vs 70% for logistic regression
- **Classical ML**: Logistic regression achieved solid 85.0% accuracy and 82.4% F1-score
- **Neural Network Advantage**: 12.3% F1-score improvement over classical ML

### Most Effective Combination
Both the **Simple Neural Network (Instance 1)** and **Optimized Neural Network (Instance 2)** proved most effective:
- Excellent generalization with 95.0% test accuracy and 94.7% F1-score
- Perfect precision (100%) with good recall (90%)
- Efficient feature learning through lightweight architecture
- Instance 2 adds L2 regularization for additional robustness
- Optimized for deployment with minimal model size

### Implementation Comparison
**Neural Networks achieved exceptional performance** due to:
1. **Feature Learning**: Automatic extraction of complex audio patterns
2. **Non-linear Modeling**: Better capture of speech disorder characteristics
3. **Hierarchical Representations**: Multi-level feature abstraction
4. **Surprising Simplicity**: Simple architectures performed as well as complex ones

**Classical ML Performance**:
- **Logistic Regression**: 85.0% accuracy, 82.4% F1-score with optimized hyperparameters
- **Perfect Precision**: Achieved 100% precision like the best neural networks
- **Lower Recall**: 70% recall vs 90% for best neural networks
- **Advantages**: Faster training, interpretable features, lower computational requirements

### Optimization Impact
- **Logistic Regression**: 82.4% F1-score (strong classical ML baseline)
- **Simple NN (Instance 1)**: 94.7% F1-score (excellent baseline)
- **Optimized Instance 2**: 94.7% F1-score (maintained performance with regularization)
- **Instance 3 (L1+BatchNorm)**: 61.5% F1-score (lower performance, possibly over-regularized)
- **Instance 4 (L1L2+Advanced)**: 88.9% F1-score (good but not best)
- **Key Insight**: Sometimes simpler architectures work better than complex optimizations

### Notable Findings
- **Instance 3 Underperformance**: The combination of L1 regularization and BatchNormalization may have been too aggressive for this dataset size, leading to underfitting
- **Regularization Balance**: L2 regularization (Instance 2) maintained performance while L1 regularization (Instance 3) caused significant performance drop
- **Architecture Complexity**: More complex architectures don't always yield better results
- **Early Stopping Effectiveness**: Applied in Instances 2, 3, and 4 with varying success

## File Structure
```
Dysarthric_Speech_Classification/
├── README.md                           # Project documentation
├── dysarthria2.ipynb                   # Main implementation notebook
├── saved_models/                       # Trained model files (lightweight versions)
│   ├── optimized_logistic_regression.pkl
│   ├── simple_neural_network.h5
│   ├── optimized_nn_instance2.h5
│   ├── optimized_nn_instance3.h5
│   ├── advanced_optimized_nn_instance4.h5
│   └── model_sizes_summary.csv         # Model compression summary
├── training_results_table.csv          # Detailed results table
├── training_comparison.png             # Training history plots
├── model_comparison.png                # Performance comparison charts
└── final_confusion_matrix.png          # Best model confusion matrix
```

## Future Work
- Implement attention mechanisms for sequence modeling
- Explore transformer architectures for speech classification
- Investigate cross-speaker generalization
- Develop real-time classification system
- Expand to multi-class severity classification
- Scale models for larger datasets while maintaining GitHub compatibility

## Model Optimization Notes
This project implements lightweight neural network architectures specifically optimized for:
- **GitHub Deployment**: All models <10MB each, total <50MB
- **Academic Requirements**: Demonstrates optimization techniques effectively
- **Performance Excellence**: Achieved 95% accuracy and 94.7% F1-score
- **Reproducibility**: Easy to run on various platforms including Google Colab
- **Surprising Insights**: Simple architectures can outperform complex optimizations

## References
- TORGO Database: University of Toronto
- TensorFlow/Keras framework
- Scikit-learn machine learning library
- Librosa audio processing library

## License
This project is licensed under the MIT License.

## Contact
For questions or collaboration, please open an issue in this repository.