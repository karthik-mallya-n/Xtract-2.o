"""
Test script to verify that the model training system works with realistic training times
"""

import sys
import os

# Add the my_flask_app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'my_flask_app'))

from core_ml import MLCore
import pandas as pd
import numpy as np

# Create a test dataset
def create_test_dataset():
    """Create a realistic test dataset for model training"""
    np.random.seed(42)
    
    # Create 1000 samples with mixed data types
    n_samples = 1000
    
    data = {
        'numeric_1': np.random.normal(0, 1, n_samples),
        'numeric_2': np.random.uniform(-10, 10, n_samples),
        'numeric_3': np.random.exponential(2, n_samples),
        'category_1': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'category_2': np.random.choice(['Type1', 'Type2', 'Type3'], n_samples),
        'boolean_feature': np.random.choice([True, False], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create a target variable for classification
    # Make it somewhat dependent on the features for realistic learning
    df['target'] = (
        (df['numeric_1'] > 0).astype(int) +
        (df['numeric_2'] > 0).astype(int) +
        (df['category_1'].isin(['A', 'B'])).astype(int) +
        np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # Add some noise
    )
    df['target'] = df['target'].clip(0, 1)  # Binary classification
    
    return df

def test_model_training():
    """Test the model training system"""
    print("🚀 TESTING MODEL-SPECIFIC TRAINING SYSTEM")
    print("="*60)
    
    # Create test dataset
    print("\n📊 Creating test dataset...")
    df = create_test_dataset()
    test_file = "test_dataset.csv"
    df.to_csv(test_file, index=False)
    print(f"✅ Test dataset created: {df.shape} samples")
    print(f"   Target distribution: {df['target'].value_counts().to_dict()}")
    
    # Initialize ML Core
    ml_core = MLCore()
    
    # Test models to train
    test_models = [
        "Random Forest",
        "Logistic Regression", 
        "Support Vector Machine",
        "Decision Tree"
    ]
    
    results = []
    
    for model_name in test_models:
        print(f"\n{'='*60}")
        print(f"🤖 TESTING MODEL: {model_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Train the model using the specific training method
            result = ml_core.train_specific_model(
                file_path=test_file,
                model_name=model_name,
                user_data={'target_column': 'target'},
                target_column='target'
            )
            
            if result['success']:
                print(f"✅ {model_name} training SUCCESS")
                print(f"   Training time: {result.get('training_time', 'Unknown')}")
                print(f"   Model performance: {result.get('performance', {})}")
                results.append({
                    'model': model_name,
                    'success': True,
                    'training_time': result.get('training_time'),
                    'performance': result.get('performance')
                })
            else:
                print(f"❌ {model_name} training FAILED: {result.get('error', 'Unknown error')}")
                results.append({
                    'model': model_name,
                    'success': False,
                    'error': result.get('error')
                })
                
        except Exception as e:
            print(f"❌ {model_name} training EXCEPTION: {str(e)}")
            results.append({
                'model': model_name,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 TRAINING SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✅ Successful trainings: {len(successful)}")
    print(f"❌ Failed trainings: {len(failed)}")
    
    if successful:
        print(f"\n🏆 SUCCESSFUL MODELS:")
        for result in successful:
            training_time = result.get('training_time', 'Unknown')
            print(f"   {result['model']}: {training_time}")
    
    if failed:
        print(f"\n💥 FAILED MODELS:")
        for result in failed:
            print(f"   {result['model']}: {result.get('error', 'Unknown error')}")
    
    # Cleanup
    try:
        os.remove(test_file)
        print(f"\n🧹 Cleaned up test file: {test_file}")
    except:
        pass

if __name__ == "__main__":
    test_model_training()