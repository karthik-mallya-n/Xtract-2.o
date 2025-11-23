#!/usr/bin/env python3
"""
Direct test of core_ml training functions to verify metrics fix
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the Flask app directory to the Python path
sys.path.insert(0, r'e:\New Codes\MP 2.o\02\my_flask_app')

from core_ml import MLCore

def create_test_data():
    """Create simple test datasets"""
    np.random.seed(42)
    n_samples = 50
    
    # Regression data
    data_reg = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'target': np.random.normal(0, 1, n_samples) * 2 + 5  # Continuous target
    }
    df_reg = pd.DataFrame(data_reg)
    
    # Classification data  
    data_class = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'target': np.random.choice([0, 1], n_samples)  # Binary target
    }
    df_class = pd.DataFrame(data_class)
    
    return df_reg, df_class

def test_direct_regression():
    print("🧪 DIRECT REGRESSION TEST")
    print("="*50)
    
    df_reg, _ = create_test_data()
    
    # Save the regression data
    reg_file = r'e:\New Codes\MP 2.o\02\direct_reg_test.csv'
    df_reg.to_csv(reg_file, index=False)
    
    print(f"📊 Regression data shape: {df_reg.shape}")
    print(f"📊 Target stats: mean={df_reg['target'].mean():.2f}, std={df_reg['target'].std():.2f}")
    print(f"📊 Unique targets: {df_reg['target'].nunique()}")
    
    user_data = {
        'is_labeled': 'labeled',
        'data_type': 'continuous',  # Force regression
        'target_column': 'target'
    }
    
    selected_columns = ['feature_1', 'feature_2', 'target']
    
    print(f"\n🚀 Training regression model...")
    try:
        # Initialize MLCore
        ml_core = MLCore()
        
        result = ml_core.train_specific_model(
            file_path=reg_file,
            model_name='Linear Regression',
            target_column='target',
            user_data=user_data,
            selected_columns=selected_columns
        )
        
        print(f"✅ Training completed: {result.get('success', False)}")
        
        feature_info = result.get('feature_info', {})
        performance = result.get('performance', {})
        
        print(f"\n📈 RESULTS:")
        print(f"   🎯 Problem type: {feature_info.get('problem_type')}")
        print(f"   📊 Target: {feature_info.get('target_column')}")
        
        print(f"\n📊 PERFORMANCE METRICS:")
        for metric, value in performance.items():
            print(f"   📈 {metric}: {value}")
        
        # Verify regression metrics
        expected_regression_metrics = ['r2_score', 'mse', 'rmse', 'mae']
        unexpected_classification_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        print(f"\n✅ REGRESSION METRICS VERIFICATION:")
        for metric in expected_regression_metrics:
            if metric in performance:
                print(f"   ✅ {metric}: PRESENT ({performance[metric]:.4f})")
            else:
                print(f"   ❌ {metric}: MISSING")
        
        print(f"\n❌ CLASSIFICATION METRICS CHECK (should be absent):")
        for metric in unexpected_classification_metrics:
            if metric in performance:
                print(f"   ❌ {metric}: PRESENT (SHOULD NOT BE) - {performance[metric]}")
            else:
                print(f"   ✅ {metric}: Correctly absent")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_direct_classification():
    print("\n\n🧪 DIRECT CLASSIFICATION TEST")
    print("="*50)
    
    _, df_class = create_test_data()
    
    # Save the classification data
    class_file = r'e:\New Codes\MP 2.o\02\direct_class_test.csv'
    df_class.to_csv(class_file, index=False)
    
    print(f"📊 Classification data shape: {df_class.shape}")
    print(f"📊 Target distribution: {df_class['target'].value_counts().to_dict()}")
    print(f"📊 Unique targets: {df_class['target'].nunique()}")
    
    user_data = {
        'is_labeled': 'labeled',
        'data_type': 'categorical',  # Force classification
        'target_column': 'target'
    }
    
    selected_columns = ['feature_1', 'feature_2', 'target']
    
    print(f"\n🚀 Training classification model...")
    try:
        # Initialize MLCore
        ml_core = MLCore()
        
        result = ml_core.train_specific_model(
            file_path=class_file,
            model_name='Random Forest',
            target_column='target',
            user_data=user_data,
            selected_columns=selected_columns
        )
        
        print(f"✅ Training completed: {result.get('success', False)}")
        
        feature_info = result.get('feature_info', {})
        performance = result.get('performance', {})
        
        print(f"\n📈 RESULTS:")
        print(f"   🎯 Problem type: {feature_info.get('problem_type')}")
        print(f"   📊 Target: {feature_info.get('target_column')}")
        
        print(f"\n📊 PERFORMANCE METRICS:")
        metric_values = []
        for metric, value in performance.items():
            print(f"   📈 {metric}: {value}")
            metric_values.append(value)
        
        # Check if all metrics are the same (the original problem)
        if len(set(metric_values)) > 1:
            print(f"\n✅ METRICS ARE DIFFERENT (CORRECT): {metric_values}")
        else:
            print(f"\n❌ ALL METRICS SAME (PROBLEM): {metric_values}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_regression()
    test_direct_classification()