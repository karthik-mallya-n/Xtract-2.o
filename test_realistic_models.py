#!/usr/bin/env python3
"""
Test script for realistic model training implementation
Tests all the specified classification models with comprehensive logging
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import time

# Add the flask app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'my_flask_app'))

from core_ml import MLCore

def create_sample_dataset():
    """Create a sample classification dataset"""
    print("📊 Creating sample classification dataset...")
    
    # Generate a classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(20)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some categorical features for realistic preprocessing
    df['category_A'] = np.random.choice(['Type1', 'Type2', 'Type3'], size=1000)
    df['category_B'] = np.random.choice(['GroupX', 'GroupY'], size=1000)
    
    # Save to CSV
    dataset_path = 'test_classification_dataset.csv'
    df.to_csv(dataset_path, index=False)
    
    print(f"✅ Dataset saved to: {dataset_path}")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🎯 Target distribution: {df['target'].value_counts().to_dict()}")
    print(f"📋 Features: {list(df.columns)}")
    
    return dataset_path

def test_model_training(ml_core, model_name, dataset_path):
    """Test training for a specific model"""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING MODEL: {model_name}")
    print(f"{'='*80}")
    
    try:
        start_time = time.time()
        
        # Test the model training
        result = ml_core.train_specific_model(
            file_path=dataset_path,
            model_name=model_name,
            user_data={'data_type': 'categorical', 'is_labeled': 'labeled'},
            target_column='target'
        )
        
        training_time = time.time() - start_time
        
        if result['success']:
            print(f"\n✅ SUCCESS: {model_name} trained successfully!")
            print(f"⏱️  Total time: {training_time:.2f} seconds")
            print(f"🎯 Test score: {result.get('test_score', 'N/A')}")
            # Fix: Use model_folder instead of model_info.model_directory
            if 'model_folder' in result:
                print(f"📁 Model saved in: {result['model_folder']}")
            return True
        else:
            print(f"\n❌ FAILED: {model_name} training failed")
            print(f"Error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ EXCEPTION: {model_name} training failed with exception")
        print(f"Error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main test function"""
    print("🚀 STARTING REALISTIC MODEL TRAINING TESTS")
    print("="*80)
    
    # Initialize MLCore
    try:
        ml_core = MLCore()
        print("✅ MLCore initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize MLCore: {e}")
        return
    
    # Create sample dataset
    try:
        dataset_path = create_sample_dataset()
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return
    
    # Models to test (as specified by the user)
    models_to_test = [
        "random-forest-classifier",
        "xgboost-classifier", 
        "lightgbm-classifier",
        "catboost-classifier",
        "svm-classifier",
        "logistic-regression",
        "neural-network-classifier",
        "knn-classifier",
        "decision-tree-classifier",
        "gradient-boosting-classifier",
        "naive-bayes"
    ]
    
    # Test results tracking
    successful_models = []
    failed_models = []
    
    print(f"\n📋 Testing {len(models_to_test)} models...")
    
    # Test each model
    for i, model_name in enumerate(models_to_test, 1):
        print(f"\n{'='*100}")
        print(f"🧪 TEST {i}/{len(models_to_test)}: {model_name}")
        print(f"{'='*100}")
        
        success = test_model_training(ml_core, model_name, dataset_path)
        
        if success:
            successful_models.append(model_name)
        else:
            failed_models.append(model_name)
        
        # Small delay between tests
        time.sleep(1)
    
    # Final Results Summary
    print(f"\n{'='*100}")
    print("📊 FINAL TEST RESULTS")
    print(f"{'='*100}")
    print(f"✅ Successful models: {len(successful_models)}/{len(models_to_test)}")
    for model in successful_models:
        print(f"   ✅ {model}")
    
    print(f"\n❌ Failed models: {len(failed_models)}/{len(models_to_test)}")
    for model in failed_models:
        print(f"   ❌ {model}")
    
    # Overall success rate
    success_rate = (len(successful_models) / len(models_to_test)) * 100
    print(f"\n🎯 Overall Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🏆 EXCELLENT: Implementation is working very well!")
    elif success_rate >= 60:
        print("✅ GOOD: Implementation is working with some issues")
    else:
        print("⚠️ NEEDS IMPROVEMENT: Many models are failing")
    
    # Cleanup
    try:
        os.remove(dataset_path)
        print(f"🧹 Cleaned up test dataset")
    except:
        pass

if __name__ == "__main__":
    main()