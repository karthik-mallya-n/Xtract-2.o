#!/usr/bin/env python3
"""
Direct Model Implementation Test
Tests all model implementations directly without requiring Flask server
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import time
import traceback

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our core ML module
from core_ml import MLCore

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def create_test_datasets():
    """Create test datasets for classification and regression"""
    
    # Classification dataset
    X_class, y_class = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=42
    )
    
    class_df = pd.DataFrame(X_class, columns=[f'feature_{i+1}' for i in range(X_class.shape[1])])
    class_df['target'] = y_class
    
    # Regression dataset  
    X_reg, y_reg = make_regression(
        n_samples=100,
        n_features=5,
        noise=0.1,
        random_state=42
    )
    
    reg_df = pd.DataFrame(X_reg, columns=[f'feature_{i+1}' for i in range(X_reg.shape[1])])
    reg_df['target'] = y_reg
    
    return class_df, reg_df

def test_model_direct(model_name, dataset_path, problem_type):
    """Test a model directly using the MLCore class"""
    
    print(f"🧪 Testing: {model_name} ({problem_type})")
    print("=" * 60)
    
    try:
        # Create MLCore instance
        ml = MLCore()
        
        # Prepare user data dictionary
        user_data = {
            'is_labeled': True,
            'target_column': 'target',
            'problem_type': problem_type,
            'data_type': 'mixed'
        }
        
        # Train the model
        start_time = time.time()
        
        result = ml.train_specific_model(
            file_path=dataset_path,
            model_name=model_name,
            user_data=user_data,
            target_column='target'
        )
        
        training_time = time.time() - start_time
        
        # Check if training was successful
        if 'error' in result:
            print(f"❌ Training failed: {result['error']}")
            return False
        else:
            print(f"✅ Training successful!")
            if 'test_score' in result and isinstance(result['test_score'], (int, float)):
                print(f"   📊 Score: {result['test_score']:.4f}")
            else:
                print(f"   📊 Score: {result.get('test_score', 'N/A')}")
            print(f"   ⏱️  Training time: {training_time:.2f} seconds")
            print(f"   📁 Model saved: {result.get('model_path', 'N/A')}")
            return True
            
    except Exception as e:
        print(f"💥 Exception during training: {str(e)}")
        traceback.print_exc()
        return False

def main():
    print("🧪 DIRECT MODEL IMPLEMENTATION TEST")
    print("=" * 80)
    print("Testing model implementations directly (no web server required)")
    print()
    
    # Create test datasets
    print("📊 Creating test datasets...")
    class_data, reg_data = create_test_datasets()
    print(f"   • Classification dataset: {class_data.shape[0]} samples, {class_data.shape[1]-1} features")
    print(f"   • Regression dataset: {reg_data.shape[0]} samples, {reg_data.shape[1]-1} features")
    
    # Save datasets to files
    print("💾 Saving test datasets...")
    class_file = "test_classification_dataset.csv"
    reg_file = "test_regression_dataset.csv"
    class_data.to_csv(class_file, index=False)
    reg_data.to_csv(reg_file, index=False)
    print(f"   • Classification data saved to: {class_file}")
    print(f"   • Regression data saved to: {reg_file}")
    print()
    
    # Define models to test
    classification_models = [
        "Random Forest",
        "XGBoost", 
        "LightGBM",
        "CatBoost",
        "Support Vector Machine",
        "Logistic Regression",
        "Neural Network",
        "K-Neighbors",
        "Decision Tree",
        "Gradient Boosting",
        "Naive Bayes"
    ]
    
    regression_models = [
        "Random Forest Regressor",
        "XGBoost Regressor",
        "LightGBM Regressor", 
        "CatBoost Regressor",
        "Support Vector Regressor",
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "ElasticNet",
        "Gradient Boosting Regressor",
        "Neural Network Regressor"
    ]
    
    # Test results tracking
    classification_results = []
    regression_results = []
    
    # Test classification models
    print("🎯 TESTING CLASSIFICATION MODELS")
    print("=" * 80)
    
    for model_name in classification_models:
        success = test_model_direct(model_name, class_file, "classification")
        classification_results.append((model_name, success))
        print()
    
    # Test regression models
    print("🎯 TESTING REGRESSION MODELS") 
    print("=" * 80)
    
    for model_name in regression_models:
        success = test_model_direct(model_name, reg_file, "regression")
        regression_results.append((model_name, success))
        print()
    
    # Summary
    print("🎯 FINAL TEST RESULTS")
    print("=" * 80)
    
    # Classification results
    class_successful = sum(1 for _, success in classification_results if success)
    class_total = len(classification_results)
    print(f"📊 Classification Models: {class_successful}/{class_total} successful ({(class_successful/class_total*100):.1f}%)")
    
    # Regression results  
    reg_successful = sum(1 for _, success in regression_results if success)
    reg_total = len(regression_results)
    print(f"📊 Regression Models: {reg_successful}/{reg_total} successful ({(reg_successful/reg_total*100):.1f}%)")
    
    # Overall
    total_successful = class_successful + reg_successful
    total_tests = class_total + reg_total
    print(f"📈 Overall Success Rate: {total_successful}/{total_tests} ({(total_successful/total_tests*100):.1f}%)")
    print()
    
    # Failed models
    failed_models = []
    failed_models.extend([name for name, success in classification_results if not success])
    failed_models.extend([name for name, success in regression_results if not success])
    
    if failed_models:
        print("❌ Failed Models:")
        for model in failed_models:
            print(f"   • {model}")
    else:
        print("✅ All models implemented successfully!")
    
    # Cleanup
    print("\n🧹 Cleaning up test files...")
    try:
        os.remove(class_file)
        os.remove(reg_file)
        print("   ✅ Test files cleaned up")
    except Exception as e:
        print(f"   ⚠️  Warning: Could not clean up test files: {e}")
    
    print()
    print("✅ Testing completed!")

if __name__ == "__main__":
    main()