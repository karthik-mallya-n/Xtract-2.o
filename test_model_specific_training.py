"""
Test script to verify model-specific training implementation
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from my_flask_app.core_ml import MLCore

# Create synthetic datasets for testing
def create_test_datasets():
    # Classification dataset
    X_class, y_class = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_informative=8, 
        n_redundant=2, 
        n_classes=3, 
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(10)]
    df_class = pd.DataFrame(X_class, columns=feature_names)
    df_class['target'] = y_class
    
    # Save classification dataset
    df_class.to_csv('test_classification.csv', index=False)
    
    # Regression dataset
    X_reg, y_reg = make_regression(
        n_samples=1000, 
        n_features=10, 
        noise=0.1, 
        random_state=42
    )
    
    # Create DataFrame
    df_reg = pd.DataFrame(X_reg, columns=feature_names)
    df_reg['target'] = y_reg
    
    # Save regression dataset
    df_reg.to_csv('test_regression.csv', index=False)
    
    print("✅ Test datasets created:")
    print(f"📊 Classification: {df_class.shape} - 3 classes")
    print(f"📊 Regression: {df_reg.shape} - continuous target")

def test_model_specific_training():
    """Test the new model-specific training approach"""
    
    print("\n" + "="*60)
    print("🧪 TESTING MODEL-SPECIFIC TRAINING")
    print("="*60)
    
    # Initialize MLCore
    ml_core = MLCore()
    
    # Test different models with classification data
    classification_models = [
        'Random Forest',
        'Logistic Regression', 
        'Decision Tree',
        'SVM'
    ]
    
    print("\n🎯 TESTING CLASSIFICATION MODELS:")
    print("-" * 40)
    
    for model_name in classification_models:
        try:
            print(f"\n📈 Testing {model_name}...")
            
            result = ml_core.train_advanced_model(
                model_name=model_name,
                file_path='test_classification.csv',
                target_column='target'
            )
            
            if result['success']:
                accuracy = result.get('test_accuracy', 'N/A')
                print(f"✅ {model_name}: Test Accuracy = {accuracy}")
            else:
                print(f"❌ {model_name}: Failed - {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ {model_name}: Exception - {str(e)}")
    
    # Test different models with regression data
    regression_models = [
        'Random Forest Regressor',
        'Linear Regression',
        'Decision Tree Regressor'
    ]
    
    print("\n🎯 TESTING REGRESSION MODELS:")
    print("-" * 40)
    
    for model_name in regression_models:
        try:
            print(f"\n📈 Testing {model_name}...")
            
            result = ml_core.train_advanced_model(
                model_name=model_name,
                file_path='test_regression.csv',
                target_column='target'
            )
            
            if result['success']:
                r2_score = result.get('test_r2', 'N/A')
                print(f"✅ {model_name}: Test R² = {r2_score}")
            else:
                print(f"❌ {model_name}: Failed - {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ {model_name}: Exception - {str(e)}")

if __name__ == "__main__":
    # Create test datasets
    create_test_datasets()
    
    # Test model-specific training
    test_model_specific_training()
    
    print("\n" + "="*60)
    print("✅ TESTING COMPLETE!")
    print("="*60)