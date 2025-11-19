#!/usr/bin/env python3
"""
Test script to verify the enhanced training system with:
- 90%/10% train/test split
- Automatic retraining with target accuracy
- Complete dataset usage verification
"""

import requests
import json
import time
import os

# Test configuration
FLASK_URL = "http://localhost:5000"
TEST_MODEL = "Lasso Regression"  # Simple model for quick testing
DATASET_FILE = "walmart_sample.csv"

def upload_dataset():
    """Upload the Walmart dataset and return file_id"""
    print("📤 Uploading Walmart dataset...")
    
    try:
        # Check if file exists
        if not os.path.exists(DATASET_FILE):
            print(f"❌ Dataset file not found: {DATASET_FILE}")
            return None
            
        # Upload file with required form data
        with open(DATASET_FILE, 'rb') as f:
            files = {'file': f}
            data = {
                'is_labeled': 'labeled',
                'data_type': 'continuous'
            }
            response = requests.post(f"{FLASK_URL}/api/upload", files=files, data=data, timeout=30)
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            file_id = result.get('file_id')
            print(f"   ✅ File uploaded successfully")
            print(f"   File ID: {file_id}")
            print(f"   Rows: {result.get('rows', 'N/A')}")
            print(f"   Columns: {result.get('columns', 'N/A')}")
            return file_id
        else:
            print(f"   ❌ Upload failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"   ❌ Upload error: {str(e)}")
        return None

def test_enhanced_training(file_id):
    """Test the enhanced training system"""
    print("\n🚀 TESTING ENHANCED TRAINING SYSTEM")
    print("=" * 50)
    
    try:
        # Test training with the enhanced system
        print(f"🤖 Training {TEST_MODEL} with enhanced system...")
        
        train_data = {
            'file_id': file_id,
            'model_name': TEST_MODEL
        }
        
        print(f"   Request: {train_data}")
        
        # Send training request
        start_time = time.time()
        response = requests.post(f"{FLASK_URL}/api/train", 
                               json=train_data, 
                               timeout=300)  # 5 minute timeout
        end_time = time.time()
        
        print(f"\n📊 TRAINING RESULTS:")
        print(f"   Duration: {end_time - start_time:.2f} seconds")
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Success: {result.get('success', False)}")
            
            # Print full result for debugging
            print(f"\n📋 FULL RESULT:")
            print(json.dumps(result, indent=2))
            
            # Check if there's a nested result
            nested_result = result.get('result', {})
            if nested_result and isinstance(nested_result, dict):
                print(f"\n📋 NESTED RESULT:")
                print(json.dumps(nested_result, indent=2))
                
                # Check enhanced features in nested result
                if nested_result.get('success'):
                    if 'train_test_split' in nested_result:
                        print(f"   Train/Test Split: {nested_result['train_test_split']}")
                    
                    if 'retrain_attempts' in nested_result:
                        print(f"   Retraining Attempts: {nested_result['retrain_attempts']}")
                        
                    if 'target_accuracy' in nested_result:
                        print(f"   Target Accuracy: {nested_result['target_accuracy']*100}%")
                        
                    if 'accuracy_achieved' in nested_result:
                        status = "✅ ACHIEVED" if nested_result['accuracy_achieved'] else "❌ NOT ACHIEVED"
                        print(f"   Accuracy Target: {status}")
                else:
                    print(f"   ❌ Training error: {nested_result.get('error', 'Unknown error')}")
            
            # Check enhanced features (original logic)
            if 'train_test_split' in result:
                print(f"   Train/Test Split: {result['train_test_split']}")
            
            if 'retrain_attempts' in result:
                print(f"   Retraining Attempts: {result['retrain_attempts']}")
                
            if 'target_accuracy' in result:
                print(f"   Target Accuracy: {result['target_accuracy']*100}%")
                
            if 'accuracy_achieved' in result:
                status = "✅ ACHIEVED" if result['accuracy_achieved'] else "❌ NOT ACHIEVED"
                print(f"   Accuracy Target: {status}")
            
            if 'dataset_info' in result:
                dataset_info = result['dataset_info']
                print(f"\n📊 DATASET USAGE:")
                print(f"   Total Rows: {dataset_info.get('total_rows', 'N/A')}")
                print(f"   Training Rows: {dataset_info.get('training_rows', 'N/A')} (90%)")
                print(f"   Test Rows: {dataset_info.get('test_rows', 'N/A')} (10%)")
                print(f"   Features: {dataset_info.get('features_count', 'N/A')}")
                
                # Verify 90/10 split
                if 'training_rows' in dataset_info and 'test_rows' in dataset_info:
                    total = dataset_info['training_rows'] + dataset_info['test_rows']
                    train_pct = (dataset_info['training_rows'] / total) * 100
                    test_pct = (dataset_info['test_rows'] / total) * 100
                    print(f"   Split Verification: {train_pct:.1f}% / {test_pct:.1f}%")
            
            if 'performance' in result:
                perf = result['performance']
                print(f"\n📈 PERFORMANCE:")
                if 'r2_score' in perf:
                    print(f"   R² Score: {perf['r2_score']:.4f} ({perf['r2_score']*100:.2f}%)")
                if 'rmse' in perf:
                    print(f"   RMSE: {perf['rmse']:.4f}")
                if 'cv_r2' in perf:
                    print(f"   CV R²: {perf['cv_r2']:.4f} ± {perf.get('cv_std', 0)*2:.4f}")
            
            print(f"\n🎯 ENHANCED FEATURES VERIFIED:")
            features = []
            
            # Check in both main result and nested result
            train_test_split = result.get('train_test_split') or nested_result.get('train_test_split')
            retrain_attempts = result.get('retrain_attempts') or nested_result.get('retrain_attempts')
            target_accuracy = result.get('target_accuracy') or nested_result.get('target_accuracy')
            
            if train_test_split == '90/10':
                features.append("✅ 90%/10% split")
            else:
                features.append("❌ 90%/10% split")
                
            if retrain_attempts is not None:
                features.append("✅ Retraining logic")
            else:
                features.append("❌ Retraining logic")
                
            if target_accuracy is not None:
                features.append("✅ Target accuracy")
            else:
                features.append("❌ Target accuracy")
                
            for feature in features:
                print(f"   {feature}")
            
            return nested_result.get('success', False) if nested_result else True
                
        else:
            print(f"❌ Training failed: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Training request timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_prediction(file_id):
    """Test prediction with enhanced model"""
    print(f"\n🔮 TESTING PREDICTIONS...")
    
    try:
        # Test prediction with correct input_data format
        prediction_data = {
            'file_id': file_id,
            'input_data': {
                'Store': 1,
                'Date': '2010-02-05',
                'Holiday_Flag': 0,
                'Temperature': 42.31,
                'Fuel_Price': 2.572,
                'CPI': 211.096,
                'Unemployment': 8.106
            }
        }
        
        response = requests.post(f"{FLASK_URL}/api/predict", 
                               json=prediction_data, 
                               timeout=30)
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Prediction: {result.get('prediction', 'N/A')}")
            print(f"   Model Used: {result.get('model_info', {}).get('model_name', 'N/A')}")
        else:
            print(f"   ❌ Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")

if __name__ == "__main__":
    print("🧪 ENHANCED TRAINING SYSTEM TEST")
    print("Testing 90%/10% split with automatic retraining")
    print("=" * 60)
    
    # Step 1: Upload dataset
    file_id = upload_dataset()
    
    if file_id:
        # Step 2: Test enhanced training
        training_success = test_enhanced_training(file_id)
        
        # Step 3: Test prediction if training succeeded
        if training_success:
            test_prediction(file_id)
    else:
        print("❌ Cannot proceed without uploaded dataset")
    
    print(f"\n✅ TEST COMPLETED")