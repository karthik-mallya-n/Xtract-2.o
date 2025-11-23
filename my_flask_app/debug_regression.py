#!/usr/bin/env python3
"""
Debug script to test the regression training logic
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os

print("🔍 DEBUGGING REGRESSION TRAINING")
print("="*50)

# Load the data
print("1. Loading data...")
df = pd.read_csv("../sample_data.csv")
print(f"   Dataset shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# Prepare the data
print("\n2. Preparing features and target...")
target_column = "performance_rating"
X = df.drop(columns=[target_column])
y = df[target_column]

print(f"   Target column: {target_column}")
print(f"   Target dtype: {y.dtype}")
print(f"   Target unique values: {y.nunique()}")
print(f"   Actual values: {sorted(y.unique())}")

# Check data types
print(f"\n3. Data type analysis...")
print(f"   Is target float? {y.dtype in ['float64', 'float32']}")
print(f"   Is target continuous? {'Yes' if y.dtype in ['float64', 'float32'] else 'No'}")
print(f"   Should be regression? {'Yes' if y.dtype in ['float64', 'float32'] else 'No'}")

# Test the logic used in the code
user_data = {'data_type': 'continuous'}
print(f"\n4. Testing classification detection logic...")
print(f"   user_data.get('data_type'): {user_data.get('data_type')}")
print(f"   user_data.get('data_type') == 'categorical': {user_data.get('data_type') == 'categorical'}")
print(f"   user_data.get('data_type') == 'continuous': {user_data.get('data_type') == 'continuous'}")

# Test the old logic
old_is_classification = user_data.get('data_type') == 'categorical' or len(y.unique()) < 20
print(f"   OLD LOGIC - is_classification: {old_is_classification}")

# Test the new logic
if user_data.get('data_type') == 'categorical':
    new_is_classification = True
elif user_data.get('data_type') == 'continuous':
    new_is_classification = False
else:
    # Use heuristic for auto-detection
    new_is_classification = len(y.unique()) < 20 and y.dtype in ['object', 'category']

print(f"   NEW LOGIC - is_classification: {new_is_classification}")

# Test training
print(f"\n5. Testing actual training...")
try:
    # Prepare categorical data (encode department)
    X_processed = X.copy()
    print(f"   Processing categorical columns...")
    categorical_cols = X_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X_processed[col] = pd.Categorical(X_processed[col]).codes
    
    print(f"   Processed features shape: {X_processed.shape}")
    print(f"   Feature names: {X_processed.columns.tolist()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    print(f"   Train set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"   Target train dtype: {y_train.dtype}")
    print(f"   Target test dtype: {y_test.dtype}")
    
    # Try Ridge Regression
    model = Ridge(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"   ✅ Ridge Regression SUCCESS!")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MSE: {mse:.4f}")
    
except Exception as e:
    print(f"   ❌ Training failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("🔍 DEBUG COMPLETE")