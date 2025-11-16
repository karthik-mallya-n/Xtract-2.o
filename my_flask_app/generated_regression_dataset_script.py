"""
High-Accuracy Model Training Script
Generated using Pipeline + GridSearchCV for optimal performance
Model: XGBRegressor
Scenario: Regression
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load Data
print("📄 Loading dataset...")
df = pd.read_csv('test_regression.csv')
print(f"Dataset shape: {df.shape}")

# Define Target & Features
print("\n🎯 Defining target and features...")
target_column = 'price'
columns_to_drop = ['price']

# The target variable y is the column 'price'
y = df[target_column]

# The features X are all columns except ['price']
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
model = XGBRegressor(random_state=42)

# Create pipeline that chains preprocessing and model
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Data Split
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if 'regression' == 'classification' else None
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Hyperparameter Tuning
print("\n🔍 Starting hyperparameter tuning...")
param_grid = {'model__n_estimators': [100, 200, 300], 'model__max_depth': [3, 6, 10], 'model__learning_rate': [0.01, 0.1, 0.2], 'model__subsample': [0.8, 0.9, 1.0]}

print(f"Parameter grid: {param_grid}")
print("Using 5-fold cross-validation...")

# Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=full_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
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

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'\nTest Set Evaluation:')
print(f'Mean Squared Error: {mse:.4f}')
print(f'R² Score: {r2:.4f}')

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
