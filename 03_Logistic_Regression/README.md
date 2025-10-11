# Model 3: Logistic Regression (Scikit-learn)

## Overview
Traditional machine learning approach using Logistic Regression for chest X-ray pneumonia classification.

## Architecture
- **Algorithm**: Logistic Regression
- **Preprocessing**: StandardScaler + PCA
- **Optimization**: Grid Search with Cross-Validation
- **Class Balancing**: Balanced class weights

## Key Features
- Feature scaling and normalization
- PCA dimensionality reduction (100 components)
- Grid search hyperparameter tuning
- Cross-validation (5-fold)
- Feature importance analysis
- Class balancing for imbalanced dataset

## Files
- `logistic_regression.py` - Main model implementation
- `train_logistic.py` - Training script
- `evaluate_logistic.py` - Evaluation script
- `results/` - Model results and plots

## Performance
- **Accuracy**: 83.5%
- **Precision**: 98.4%
- **Recall**: 78.7%
- **F1-Score**: 87.5%
- **Specificity**: 96.5%

## Usage
```python
from logistic_regression import LogisticRegressionClassifier

# Create classifier
lr_classifier = LogisticRegressionClassifier()

# Train model
lr_classifier.train(X_train, y_train, use_pca=True, n_components=100)

# Make predictions
predictions = lr_classifier.predict(X_test)
```

## Model Pipeline
```
Input Features (flattened images)
├── StandardScaler()  # Normalize features
├── PCA(n_components=100)  # Dimensionality reduction
└── LogisticRegression(
    C=optimized,  # Regularization strength
    penalty='l1'/'l2',  # Regularization type
    solver='liblinear'/'saga'  # Optimization algorithm
)
```

## Hyperparameter Tuning
- **C**: [0.001, 0.01, 0.1, 1, 10, 100]
- **Penalty**: ['l1', 'l2']
- **Solver**: ['liblinear', 'saga']
- **Cross-validation**: 5-fold
- **Scoring**: F1-score
