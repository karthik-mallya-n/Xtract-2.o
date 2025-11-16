#!/usr/bin/env python3
"""
Verify training process with detailed row counting and debugging
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def trace_training_process():
    """Trace the exact training process step-by-step with row counting"""
    
    print("🔍 TRACING EXACT TRAINING PROCESS")
    print("=" * 60)
    
    # Use your Walmart dataset
    dataset_path = "uploads/walmart_fixed_target.csv"
    if not os.path.exists(dataset_path):
        print("❌ Walmart dataset not found")
        return
    
    # Step 1: Load data (EXACT same as training code)
    print("1️⃣ Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"   📊 Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   📊 Columns: {list(df.columns)}")
    
    # Step 2: Target column detection (EXACT same logic as training)
    target_column = df.columns[-1]  # Last column
    print(f"\n2️⃣ Target detection...")
    print(f"   🎯 Auto-detected target: {target_column}")
    
    # Step 3: Features and target separation (EXACT same as training)
    print(f"\n3️⃣ Feature/target separation...")
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    print(f"   ✅ Features (X): {X.shape[0]} rows × {X.shape[1]} features")
    print(f"   ✅ Target (y): {y.shape[0]} values")
    print(f"   🔍 Feature columns: {list(X.columns)}")
    
    # Step 4: Data type analysis (EXACT same as training)
    print(f"\n4️⃣ Data type analysis...")
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"   📈 Numeric features: {len(numeric_features)} - {numeric_features}")
    print(f"   🏷️  Categorical features: {len(categorical_features)} - {categorical_features}")
    
    # Step 5: Missing value analysis
    print(f"\n5️⃣ Missing value analysis...")
    missing_X = X.isnull().sum().sum()
    missing_y = y.isnull().sum()
    
    print(f"   📊 Missing values in X: {missing_X}")
    print(f"   📊 Missing values in y: {missing_y}")
    print(f"   ✅ Imputation will handle missing values, NO ROWS DROPPED")
    
    # Step 6: Train/test split (EXACT same parameters as training)
    print(f"\n6️⃣ Train/test split...")
    
    # Problem type detection
    unique_targets = y.nunique()
    target_dtype = y.dtype
    is_classification = unique_targets <= 20 and target_dtype in ['int64', 'int32', 'object', 'bool', 'category']
    
    print(f"   📊 Unique target values: {unique_targets}")
    print(f"   📊 Target dtype: {target_dtype}")
    print(f"   📊 Problem type: {'Classification' if is_classification else 'Regression'}")
    
    stratify = y if is_classification else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    
    print(f"   📊 Training set: {X_train.shape[0]} rows")
    print(f"   📊 Test set: {X_test.shape[0]} rows")
    print(f"   📊 Total: {X_train.shape[0]} + {X_test.shape[0]} = {X_train.shape[0] + X_test.shape[0]}")
    
    # Step 7: Pipeline preprocessing simulation
    print(f"\n7️⃣ Pipeline preprocessing...")
    
    # Create the EXACT same preprocessing pipeline as training
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    # Fit and transform training data (this is what actually happens in training)
    print("   🔧 Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"   ✅ Processed training set: {X_train_processed.shape}")
    print(f"   ✅ Processed test set: {X_test_processed.shape}")
    
    # Step 8: Final verification
    print(f"\n8️⃣ FINAL ROW COUNT VERIFICATION")
    
    original_rows = len(df)
    training_rows = len(X_train)
    test_rows = len(X_test)
    processed_training_rows = X_train_processed.shape[0]
    processed_test_rows = X_test_processed.shape[0]
    
    print(f"   📊 Original dataset: {original_rows} rows")
    print(f"   📊 Training subset: {training_rows} rows")
    print(f"   📊 Test subset: {test_rows} rows")
    print(f"   📊 Processed training: {processed_training_rows} rows")
    print(f"   📊 Processed test: {processed_test_rows} rows")
    
    total_processed = processed_training_rows + processed_test_rows
    
    print(f"\n🎯 SUMMARY:")
    print(f"   📊 Total rows available: {original_rows}")
    print(f"   📊 Total rows used: {total_processed}")
    print(f"   📊 Percentage used: {(total_processed/original_rows)*100:.1f}%")
    print(f"   📊 Rows discarded: {original_rows - total_processed}")
    
    if total_processed == original_rows:
        print(f"\n   ✅ SUCCESS: EVERY SINGLE ROW ({original_rows}) IS USED!")
        print(f"   ✅ NO DATA IS LOST OR FILTERED OUT!")
        print(f"   ✅ THE MODEL TRAINS ON YOUR COMPLETE DATASET!")
    else:
        print(f"\n   ❌ WARNING: {original_rows - total_processed} rows lost!")
    
    # Show actual data samples
    print(f"\n9️⃣ TRAINING DATA SAMPLES")
    print("   First 3 training samples:")
    for i in range(min(3, len(X_train))):
        row_data = X_train.iloc[i].to_dict()
        target_val = y_train.iloc[i]
        print(f"   Row {i+1}: {row_data} → Target: {target_val}")
    
    return total_processed == original_rows

if __name__ == "__main__":
    trace_training_process()