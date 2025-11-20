#!/usr/bin/env python3
"""
Comprehensive Model Implementation Test
Tests all models from the frontend naming convention list to ensure they're properly configured
"""
import requests
import pandas as pd
import sys
import os
from io import StringIO

# Configuration
BACKEND_URL = "http://localhost:5000"

def create_test_datasets():
    """Create test datasets for classification, regression, and clustering"""
    
    # Classification dataset (customer churn prediction)
    classification_data = {
        'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'Age': [25, 34, 28, 45, 31, 29, 38, 42, 33, 27, 36, 44, 30, 26, 39, 35, 41, 32, 37, 43],
        'Income': [45000, 65000, 55000, 85000, 60000, 50000, 75000, 90000, 62000, 48000, 70000, 88000, 58000, 46000, 78000, 67000, 82000, 53000, 71000, 86000],
        'SpendingScore': [78, 56, 69, 42, 61, 73, 48, 35, 64, 81, 52, 41, 67, 79, 45, 59, 38, 74, 50, 43],
        'MonthsActive': [12, 24, 18, 36, 22, 16, 30, 40, 24, 14, 28, 38, 20, 12, 32, 26, 34, 19, 27, 35],
        'ProductsOwned': [2, 4, 3, 6, 5, 3, 5, 7, 4, 2, 5, 6, 3, 2, 6, 4, 7, 3, 5, 6],
        'Churn': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Target column
    }
    
    # Regression dataset (house price prediction)
    regression_data = {
        'HouseID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'Size': [1200, 1500, 1100, 2000, 1800, 1300, 1600, 2200, 1400, 1000, 1700, 1900, 1350, 1150, 2100, 1450, 2050, 1250, 1750, 1950],
        'Bedrooms': [2, 3, 2, 4, 3, 2, 3, 4, 3, 2, 3, 4, 3, 2, 4, 3, 4, 2, 3, 4],
        'Bathrooms': [1, 2, 1, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 3, 2, 3, 1, 2, 3],
        'Age': [10, 5, 15, 3, 8, 12, 7, 2, 9, 18, 6, 4, 11, 16, 1, 10, 3, 14, 7, 5],
        'Location': ['Urban', 'Suburban', 'Urban', 'Rural', 'Suburban', 'Urban', 'Suburban', 'Rural', 'Urban', 'Suburban', 'Rural', 'Urban', 'Suburban', 'Urban', 'Rural', 'Suburban', 'Rural', 'Urban', 'Suburban', 'Rural'],
        'Price': [250000, 320000, 230000, 480000, 380000, 270000, 350000, 520000, 310000, 200000, 410000, 460000, 290000, 240000, 550000, 330000, 500000, 260000, 390000, 470000]  # Target column
    }
    
    classification_df = pd.DataFrame(classification_data)
    regression_df = pd.DataFrame(regression_data)
    
    return classification_df, regression_df

def test_model_training(model_name, dataset_type, dataset_csv, target_column):
    """Test training a specific model"""
    print(f"\n🧪 Testing: {model_name} ({dataset_type})")
    print("=" * 60)
    
    try:
        # Upload dataset
        files = {'file': (f'{dataset_type}_dataset.csv', StringIO(dataset_csv), 'text/csv')}
        data = {
            'is_labeled': 'true',
            'data_type': 'mixed'
        }
        
        upload_response = requests.post(
            f"{BACKEND_URL}/api/upload",
            files=files,
            data=data
        )
        
        if upload_response.status_code != 200:
            print(f"❌ Upload failed: {upload_response.text}")
            return False
            
        upload_data = upload_response.json()
        file_id = upload_data['file_id']
        print(f"✅ Dataset uploaded (ID: {file_id[:8]}...)")
        
        # Train model
        training_payload = {
            'file_id': file_id,
            'model_name': model_name,
            'target_column': target_column
        }
        
        training_response = requests.post(
            f"{BACKEND_URL}/api/train-specific-model",
            json=training_payload,
            headers={'Content-Type': 'application/json'},
            timeout=300  # 5 minutes timeout for training
        )
        
        if training_response.status_code != 200:
            print(f"❌ Training failed: {training_response.text}")
            return False
            
        training_data = training_response.json()
        
        if training_data.get('success'):
            model_info = training_data.get('model_info', {})
            performance = training_data.get('performance', {})
            
            print(f"✅ Training successful!")
            print(f"   📊 Model: {model_info.get('name', 'N/A')}")
            print(f"   📊 Type: {model_info.get('type', 'N/A')}")
            print(f"   📊 Score: {performance.get('accuracy', 'N/A')}")
            print(f"   ⏱️ Time: {model_info.get('training_time', 'N/A'):.2f}s")
            
            return True
        else:
            print(f"❌ Training failed: {training_data.get('error', 'Unknown error')}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏰ Training timeout (>5 minutes) - Model may be too complex")
        return False
    except Exception as e:
        print(f"💥 Test failed with exception: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🧪 COMPREHENSIVE MODEL IMPLEMENTATION TEST")
    print("=" * 80)
    print("Testing all models from the frontend naming convention list...")
    
    # Create test datasets
    classification_df, regression_df = create_test_datasets()
    classification_csv = classification_df.to_csv(index=False)
    regression_csv = regression_df.to_csv(index=False)
    
    # All models from your list
    models_to_test = {
        # Classification Models
        'Classification': {
            'dataset': classification_csv,
            'target': 'Churn',
            'models': [
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
        },
        
        # Regression Models
        'Regression': {
            'dataset': regression_csv,
            'target': 'Price',
            'models': [
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
        }
        
        # Note: Clustering models (KMeans, DBSCAN, Hierarchical) will be tested separately
        # as they don't require labeled data
    }
    
    # Track results
    total_tests = 0
    successful_tests = 0
    failed_models = []
    
    # Test each model category
    for category, config in models_to_test.items():
        print(f"\n{'='*80}")
        print(f"🎯 TESTING {category.upper()} MODELS")
        print(f"{'='*80}")
        
        for model_name in config['models']:
            total_tests += 1
            
            success = test_model_training(
                model_name=model_name,
                dataset_type=category.lower(),
                dataset_csv=config['dataset'],
                target_column=config['target']
            )
            
            if success:
                successful_tests += 1
            else:
                failed_models.append(f"{model_name} ({category})")
    
    # Final Results
    print(f"\n{'='*80}")
    print(f"🎯 FINAL TEST RESULTS")
    print(f"{'='*80}")
    print(f"📊 Total tests: {total_tests}")
    print(f"✅ Successful: {successful_tests}")
    print(f"❌ Failed: {len(failed_models)}")
    print(f"📈 Success rate: {(successful_tests/total_tests)*100:.1f}%")
    
    if failed_models:
        print(f"\n❌ Failed Models:")
        for model in failed_models:
            print(f"   • {model}")
    else:
        print(f"\n🎉 ALL MODELS IMPLEMENTED SUCCESSFULLY!")
        
    print(f"\n📝 Note: Clustering models (KMeans, DBSCAN, Hierarchical) require")
    print(f"    separate testing as they don't use labeled data.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        sys.exit(1)