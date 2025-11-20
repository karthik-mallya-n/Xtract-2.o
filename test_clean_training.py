"""
Test the clean ML implementation with realistic training times
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the my_flask_app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'my_flask_app'))

from core_ml_clean import CleanMLCore

def create_test_dataset(n_samples=1000):
    """Create a realistic test dataset"""
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_samples),
        'experience': np.random.randint(0, 40, n_samples),
        'score': np.random.uniform(0, 100, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target based on features (for realistic learning)
    df['approved'] = (
        (df['income'] > 45000).astype(int) +
        (df['age'] > 25).astype(int) +
        (df['education'].isin(['Master', 'PhD'])).astype(int) +
        np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    )
    df['approved'] = (df['approved'] >= 2).astype(int)  # Binary classification
    
    return df

def test_comprehensive_training():
    """Test comprehensive model training"""
    
    print("🚀 TESTING CLEAN ML IMPLEMENTATION")
    print("="*80)
    
    # Create test dataset
    df = create_test_dataset(1500)  # Larger dataset for realistic training times
    test_file = "test_data.csv"
    df.to_csv(test_file, index=False)
    
    print(f"📊 Created test dataset: {df.shape}")
    print(f"   Target distribution: {df['approved'].value_counts().to_dict()}")
    
    # Initialize clean ML core
    ml_core = CleanMLCore()
    
    # Test different models
    test_models = [
        "Random Forest",
        "Logistic Regression", 
        "Support Vector Machine",
        "Neural Network",
        "Decision Tree"
    ]
    
    results = []
    
    for model_name in test_models:
        print(f"\n{'='*80}")
        print(f"🤖 TESTING: {model_name.upper()}")
        print(f"{'='*80}")
        
        result = ml_core.train_specific_model(
            file_path=test_file,
            model_name=model_name,
            target_column='approved'
        )
        
        results.append(result)
        
        if result['success']:
            print(f"✅ SUCCESS: {model_name}")
            print(f"   Training time: {result['training_time']}")
            print(f"   {result['score_name']}: {result['test_score']:.4f}")
        else:
            print(f"❌ FAILED: {model_name}")
            print(f"   Error: {result['error']}")
    
    # Summary
    print(f"\n{'='*80}")
    print("📋 FINAL SUMMARY")
    print(f"{'='*80}")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✅ Successful trainings: {len(successful)}/{len(results)}")
    
    if successful:
        print(f"\n🏆 SUCCESSFUL MODELS:")
        for result in successful:
            print(f"   {result['model_name']}: {result['training_time']}, {result['score_name']}: {result['test_score']:.4f}")
    
    if failed:
        print(f"\n💥 FAILED MODELS:")
        for result in failed:
            print(f"   {result['model_name']}: {result['error']}")
    
    # Verify realistic training times
    print(f"\n⏱️  TRAINING TIME ANALYSIS:")
    training_times = []
    for result in successful:
        if 'seconds' in result['training_time']:
            time_str = result['training_time'].replace(' seconds', '')
            try:
                time_val = float(time_str)
                training_times.append(time_val)
                status = "✅ REALISTIC" if time_val > 3 else "⚠️  TOO FAST"
                print(f"   {result['model_name']}: {time_val:.1f}s {status}")
            except:
                print(f"   {result['model_name']}: {result['training_time']} (unparseable)")
    
    if training_times:
        avg_time = sum(training_times) / len(training_times)
        print(f"\n📊 Average training time: {avg_time:.1f} seconds")
        print(f"   Range: {min(training_times):.1f}s - {max(training_times):.1f}s")
        
        if avg_time > 5:
            print("✅ Training times are REALISTIC (>5 seconds average)")
        else:
            print("⚠️  Training times might still be too fast")
    
    # Cleanup
    try:
        os.remove(test_file)
        print(f"\n🧹 Cleaned up test file")
    except:
        pass

if __name__ == "__main__":
    test_comprehensive_training()