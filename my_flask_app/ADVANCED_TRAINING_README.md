# 🚀 Advanced Model Training System

## Overview

The Advanced Model Training System provides high-performance machine learning model training with **90%+ accuracy optimization**. Each model is automatically stored in its own dedicated folder with comprehensive metadata and preprocessing pipelines.

## 🎯 Key Features

### ✅ **90%+ Accuracy Optimization**
- Advanced hyperparameter tuning with GridSearchCV
- Feature selection and engineering
- Robust data preprocessing
- Cross-validation for reliable performance estimates

### 📁 **Organized Model Storage**
- Each model gets its own folder (e.g., `models/logistic_regression/`)
- Automatic timestamping for version control
- Complete metadata and performance tracking

### 🤖 **Comprehensive Model Support**

#### Classification Models:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM)
- K-Neighbors Classifier
- Decision Tree Classifier
- Naive Bayes
- Neural Network (MLP)
- XGBoost Classifier (if available)
- LightGBM Classifier (if available)

#### Regression Models:
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
- K-Neighbors Regressor
- Decision Tree Regressor
- Neural Network Regressor (MLP)
- XGBoost Regressor (if available)
- LightGBM Regressor (if available)

## 🔧 Installation

### 1. Install Required Packages
```bash
cd my_flask_app
pip install -r requirements.txt
```

### 2. Install Optional High-Performance Libraries
```bash
# For maximum accuracy (highly recommended)
pip install xgboost lightgbm
```

## 🚀 Quick Start

### 1. Start the Flask Server
```bash
cd my_flask_app
python app.py
```

### 2. Use the Advanced Training API

#### Get Available Models
```bash
curl "http://localhost:5000/api/available-models"
```

#### Train a Model
```bash
curl -X POST "http://localhost:5000/api/train-advanced" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "your-uploaded-file-id",
    "model_name": "Random Forest",
    "target_column": "your_target_column"
  }'
```

## 📋 API Endpoints

### `/api/train-advanced` (POST)
Train a model with 90%+ accuracy optimization

**Request Body:**
```json
{
  "file_id": "uuid-of-uploaded-file",
  "model_name": "Logistic Regression",
  "target_column": "target_column_name"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Advanced model training completed successfully - Achieved 90%+ Accuracy!",
  "model_name": "Logistic Regression",
  "model_folder": "models/logistic_regression",
  "performance": {
    "accuracy": 0.9234,
    "cv_accuracy": 0.9156,
    "classification_report": {...}
  },
  "main_score": 0.9234,
  "score_name": "Accuracy",
  "problem_type": "classification",
  "threshold_met": true
}
```

### `/api/available-models` (GET)
Get list of available models

**Query Parameters:**
- `problem_type` (optional): 'classification' or 'regression'

**Response:**
```json
{
  "success": true,
  "models": [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    ...
  ],
  "total_count": 12
}
```

## 📁 Model Folder Structure

Each trained model creates a dedicated folder:

```
models/
├── logistic_regression/
│   ├── logistic_regression_20241101_143022.pkl     # Trained model
│   ├── preprocessing_20241101_143022.pkl           # Preprocessing pipeline
│   └── metadata_20241101_143022.json               # Model metadata
├── random_forest/
│   ├── random_forest_20241101_144015.pkl
│   ├── preprocessing_20241101_144015.pkl
│   └── metadata_20241101_144015.json
└── ...
```

### Metadata File Contents:
```json
{
  "model_name": "Logistic Regression",
  "problem_type": "classification",
  "target_column": "promoted",
  "feature_names": ["age", "salary", "experience"],
  "training_date": "2024-11-01T14:30:22",
  "dataset_info": {
    "total_samples": 1000,
    "features": 8,
    "target_unique_values": 2
  },
  "performance": {
    "accuracy": 0.9234,
    "cv_accuracy": 0.9156,
    "cv_std": 0.0123
  },
  "main_score": 0.9234,
  "score_name": "Accuracy"
}
```

## 🧪 Testing

### Run the Complete Test Suite
```bash
cd my_flask_app

# Test all functionality
python test_advanced_training.py

# Or use the batch script (Windows)
test_advanced_system.bat
```

### Manual Testing Steps

1. **Upload a dataset:**
   ```bash
   curl -X POST "http://localhost:5000/api/upload" \
     -F "file=@your_dataset.csv" \
     -F "is_labeled=labeled" \
     -F "data_type=categorical"
   ```

2. **Get AI recommendations:**
   ```bash
   curl "http://localhost:5000/api/recommend-model?file_id=YOUR_FILE_ID"
   ```

3. **Train with advanced system:**
   ```bash
   curl -X POST "http://localhost:5000/api/train-advanced" \
     -H "Content-Type: application/json" \
     -d '{
       "file_id": "YOUR_FILE_ID",
       "model_name": "Random Forest",
       "target_column": "your_target"
     }'
   ```

## 🎯 Performance Optimization Features

### 1. **Advanced Preprocessing**
- Automatic missing value handling
- Smart categorical encoding
- Feature scaling with RobustScaler
- Feature selection for high-dimensional data

### 2. **Hyperparameter Tuning**
- GridSearchCV with 5-fold cross-validation
- Optimized parameter grids for each model
- Automatic best parameter selection

### 3. **Model Validation**
- Cross-validation scoring
- Separate test set evaluation
- Performance metrics tracking

### 4. **Data Quality Enhancement**
- Outlier handling
- Feature importance analysis
- Automatic problem type detection

## 🔍 Troubleshooting

### Model Training Fails
1. Check if target column exists in dataset
2. Ensure sufficient data samples (>50 recommended)
3. Verify no missing target values
4. Check data types are appropriate

### Below 90% Accuracy
1. Try different models (Random Forest, Gradient Boosting work well)
2. Ensure data quality is good
3. Consider feature engineering
4. Increase dataset size if possible

### XGBoost/LightGBM Not Available
```bash
# Install optional high-performance libraries
pip install xgboost lightgbm
```

## 📊 Expected Performance

### Classification Datasets:
- **Simple datasets:** 85-95% accuracy
- **Complex datasets:** 80-90% accuracy
- **High-quality datasets:** 90-98% accuracy

### Regression Datasets:
- **Simple datasets:** 85-95% R² score
- **Complex datasets:** 75-90% R² score
- **High-quality datasets:** 90-98% R² score

## 🎉 Success Stories

The system is designed to achieve **90%+ accuracy** on most well-structured datasets through:

1. **Intelligent preprocessing** that handles real-world data issues
2. **Comprehensive hyperparameter tuning** for optimal performance
3. **Advanced ensemble methods** (Random Forest, Gradient Boosting)
4. **Cross-validation** for reliable performance estimates
5. **Feature selection** to focus on the most important variables

**Ready to train high-performance models! 🚀**