"""
High-Accuracy Model Training Script
Generated using Pipeline + GridSearchCV for optimal performance
Model: LGBMClassifier
Scenario: Classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier

# Load Data
print("📄 Loading dataset...")
df = pd.read_csv('advanced_test.csv')
print(f"Dataset shape: {df.shape}")

# Define Target & Features
print("\n🎯 Defining target and features...")
target_column = 'target'
columns_to_drop = ['target', 'id_column', 'timestamp']

# The target variable y is the column 'target'
y = df[target_column]

# The features X are all columns except ['target', 'id_column', 'timestamp']
feature_columns = [col for col in df.columns if col not in columns_to_drop]
X = df[feature_columns]

print(f"Target variable: {target_column}")
print(f"Number of features: {len(feature_columns)}")
print(f"Features: {feature_columns}")

# Automatic Preprocessing
print("\n🔄 Setting up preprocessing pipeline...")

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numerical features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# Create preprocessing pipelines
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Create Full Pipeline
print("\n🤖 Creating full training pipeline...")
model = LGBMClassifier(random_state=42, verbose=-1)

# Create pipeline that chains preprocessing and model
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Data Split
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if 'classification' == 'classification' else None
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Hyperparameter Tuning
print("\n🔍 Starting hyperparameter tuning...")
param_grid = {'model__n_estimators': [100, 200, 300], 'model__num_leaves': [31, 50, 100], 'model__learning_rate': [0.05, 0.1, 0.15], 'model__feature_fraction': [0.8, 0.9, 1.0]}

print(f"Parameter grid: {param_grid}")
print("Using 5-fold cross-validation...")

# Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=full_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Train & Evaluate
print("\n🚀 Training model with grid search...")
grid_search.fit(X_train, y_train)

print("\n🏆 TRAINING COMPLETED!")
print("=" * 50)

# Best parameters
print("Best Tuned Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

# Best cross-validation score
print(f"\nBest Cross-Validation Score: {grid_search.best_score_:.4f}")

# Test set evaluation
print("\n📈 Evaluating on test set...")
y_pred = grid_search.predict(X_test)

print('\nTest Set Evaluation:')
print(classification_report(y_test, y_pred))

# Feature importance (if available)
try:
    if hasattr(grid_search.best_estimator_.named_steps['model'], 'feature_importances_'):
        feature_names = (numeric_features + 
                        list(grid_search.best_estimator_.named_steps['preprocessor']
                             .named_transformers_['cat']
                             .named_steps['encoder']
                             .get_feature_names_out(categorical_features)))
        
        importances = grid_search.best_estimator_.named_steps['model'].feature_importances_
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("\n📊 Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"  {i+1:2d}. {feature:30s} {importance:.4f}")
except:
    print("\nFeature importance not available for this model.")

print("\n✅ Training completed successfully!")
