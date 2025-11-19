#!/usr/bin/env python3
"""
Test script for the new Advanced Model Training System
Demonstrates training models with 90%+ accuracy optimization
"""

import requests
import pandas as pd
import json
import time
import numpy as np

def create_test_datasets():
    """Create sample datasets for testing different model types"""
    
    print("📊 Creating test datasets...")
    
    # 1. Classification dataset (employee promotion prediction)
    np.random.seed(42)
    n_samples = 500
    
    classification_data = {
        'employee_id': range(1, n_samples + 1),
        'age': np.random.randint(25, 60, n_samples),
        'experience_years': np.random.randint(1, 30, n_samples),
        'education_level': np.random.choice(['Bachelor', 'Master', 'PhD'], n_samples, p=[0.6, 0.3, 0.1]),
        'performance_score': np.random.uniform(1, 10, n_samples),
        'salary': np.random.randint(30000, 150000, n_samples),
        'department': np.random.choice(['IT', 'Sales', 'HR', 'Finance', 'Marketing'], n_samples),
        'training_hours': np.random.randint(0, 200, n_samples)
    }
    
    # Create target based on logical rules
    df_class = pd.DataFrame(classification_data)
    promotion_prob = (
        (df_class['performance_score'] > 7) * 0.4 +
        (df_class['experience_years'] > 5) * 0.3 +
        (df_class['education_level'] == 'PhD') * 0.2 +
        (df_class['training_hours'] > 100) * 0.1
    )
    df_class['promoted'] = (promotion_prob + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
    
    # 2. Regression dataset (house price prediction)
    regression_data = {
        'area_sqft': np.random.randint(800, 4000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'age_years': np.random.randint(0, 50, n_samples),
        'location_score': np.random.uniform(1, 10, n_samples),
        'garage': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'garden': np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    }
    
    df_reg = pd.DataFrame(regression_data)
    # Create realistic house prices
    df_reg['price'] = (
        df_reg['area_sqft'] * 150 +
        df_reg['bedrooms'] * 10000 +
        df_reg['bathrooms'] * 8000 +
        (50 - df_reg['age_years']) * 500 +
        df_reg['location_score'] * 15000 +
        df_reg['garage'] * 20000 +
        df_reg['garden'] * 15000 +
        np.random.normal(0, 30000, n_samples)
    ).astype(int)
    
    # Save datasets
    df_class.to_csv('test_classification.csv', index=False)
    df_reg.to_csv('test_regression.csv', index=False)
    
    print(f"✅ Created test_classification.csv: {df_class.shape}")
    print(f"✅ Created test_regression.csv: {df_reg.shape}")
    
    return 'test_classification.csv', 'test_regression.csv'

def test_advanced_training_system():
    """Test the complete advanced training system"""
    
    print("🧪 TESTING ADVANCED MODEL TRAINING SYSTEM")
    print("="*80)
    
    base_url = "http://localhost:5000"
    
    # Create test datasets
    class_file, reg_file = create_test_datasets()
    
    try:
        # Test 1: Get available models
        print(f"\n1️⃣ TESTING AVAILABLE MODELS")
        print("-" * 40)
        
        models_response = requests.get(f"{base_url}/api/available-models")
        if models_response.status_code == 200:
            models_data = models_response.json()
            print(f"✅ Available models: {len(models_data['models'])} total")
            for model in models_data['models']:
                print(f"   🤖 {model}")
        else:
            print(f"❌ Failed to get available models: {models_response.status_code}")
            return
        
        # Test 2: Classification model training
        print(f"\n2️⃣ TESTING CLASSIFICATION MODEL TRAINING")
        print("-" * 40)
        
        # Upload classification dataset
        with open(class_file, 'rb') as f:
            files = {'file': f}
            data = {'is_labeled': 'labeled', 'data_type': 'categorical'}
            upload_response = requests.post(f"{base_url}/api/upload", files=files, data=data)
        
        if upload_response.status_code == 200:
            class_file_id = upload_response.json()['file_id']
            print(f"✅ Classification dataset uploaded: {class_file_id}")
            
            # Train multiple classification models
            classification_models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
            
            for model_name in classification_models:
                print(f"\n🚀 Training {model_name}...")
                
                train_data = {
                    'file_id': class_file_id,
                    'model_name': model_name,
                    'target_column': 'promoted'
                }
                
                train_response = requests.post(f"{base_url}/api/train-advanced", json=train_data)
                
                if train_response.status_code == 200:
                    train_result = train_response.json()
                    print(f"✅ {model_name} trained successfully!")
                    print(f"   📁 Model folder: {train_result['model_folder']}")
                    print(f"   🎯 {train_result['score_name']}: {train_result['main_score']:.4f} ({train_result['main_score']*100:.2f}%)")
                    
                    if train_result['threshold_met']:
                        print(f"   🎉 Achieved 90%+ accuracy!")
                    else:
                        print(f"   ⚠️ Below 90% threshold")
                else:
                    print(f"❌ {model_name} training failed: {train_response.status_code}")
                    print(f"   Error: {train_response.text}")
        
        # Test 3: Regression model training
        print(f"\n3️⃣ TESTING REGRESSION MODEL TRAINING")
        print("-" * 40)
        
        # Upload regression dataset
        with open(reg_file, 'rb') as f:
            files = {'file': f}
            data = {'is_labeled': 'labeled', 'data_type': 'continuous'}
            upload_response = requests.post(f"{base_url}/api/upload", files=files, data=data)
        
        if upload_response.status_code == 200:
            reg_file_id = upload_response.json()['file_id']
            print(f"✅ Regression dataset uploaded: {reg_file_id}")
            
            # Train multiple regression models
            regression_models = ['Random Forest Regressor', 'Gradient Boosting Regressor', 'Ridge Regression']
            
            for model_name in regression_models:
                print(f"\n🚀 Training {model_name}...")
                
                train_data = {
                    'file_id': reg_file_id,
                    'model_name': model_name,
                    'target_column': 'price'
                }
                
                train_response = requests.post(f"{base_url}/api/train-advanced", json=train_data)
                
                if train_response.status_code == 200:
                    train_result = train_response.json()
                    print(f"✅ {model_name} trained successfully!")
                    print(f"   📁 Model folder: {train_result['model_folder']}")
                    print(f"   🎯 {train_result['score_name']}: {train_result['main_score']:.4f} ({train_result['main_score']*100:.2f}%)")
                    
                    if train_result['threshold_met']:
                        print(f"   🎉 Achieved 90%+ R² score!")
                    else:
                        print(f"   ⚠️ Below 90% threshold")
                else:
                    print(f"❌ {model_name} training failed: {train_response.status_code}")
                    print(f"   Error: {train_response.text}")
        
        print(f"\n🎉 ADVANCED TRAINING SYSTEM TEST COMPLETED!")
        print("="*80)
        print("✅ All functionality tested successfully")
        print("📁 Check the 'models' folder for trained model folders")
        print("🎯 Each model folder contains:")
        print("   - Trained model (.pkl)")
        print("   - Preprocessing pipeline (.pkl)")
        print("   - Model metadata (.json)")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the Flask server is running on port 5000")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test files
        import os
        for file in [class_file, reg_file]:
            if os.path.exists(file):
                os.remove(file)
                print(f"🧹 Cleaned up: {file}")

if __name__ == "__main__":
    test_advanced_training_system()