"""
Test the realistic ML implementation with truly comprehensive parameter grids
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the my_flask_app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'my_flask_app'))

from core_ml_realistic import RealisticMLCore

def create_comprehensive_dataset(n_samples=2000):
    """Create a comprehensive, realistic dataset for thorough testing"""
    np.random.seed(42)
    
    data = {
        # Numeric features with different distributions
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10.5, 0.8, n_samples),  # Log-normal distribution
        'experience_years': np.random.randint(0, 40, n_samples),
        'credit_score': np.random.normal(700, 100, n_samples).clip(300, 850),
        'debt_ratio': np.random.beta(2, 5, n_samples),  # Beta distribution
        'savings_amount': np.random.exponential(10000, n_samples),
        
        # Categorical features
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD', 'Associate'], 
                                    n_samples, p=[0.3, 0.4, 0.2, 0.05, 0.05]),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                                'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'], 
                               n_samples),
        'job_category': np.random.choice(['Technology', 'Finance', 'Healthcare', 'Education', 
                                        'Manufacturing', 'Retail', 'Government', 'Other'], 
                                       n_samples, p=[0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], 
                                         n_samples, p=[0.4, 0.45, 0.1, 0.05]),
        
        # Boolean/binary features
        'owns_home': np.random.choice([True, False], n_samples, p=[0.6, 0.4]),
        'has_car': np.random.choice([True, False], n_samples, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Create a complex target variable based on multiple features
    # This makes the learning problem realistic and challenging
    approval_score = (
        # Income factor (normalized)
        (df['income'] > df['income'].median()).astype(int) * 3 +
        
        # Age factor
        ((df['age'] >= 25) & (df['age'] <= 65)).astype(int) * 2 +
        
        # Education factor
        df['education'].map({'PhD': 3, 'Master': 2, 'Bachelor': 2, 'Associate': 1, 'High School': 0}) +
        
        # Credit score factor
        (df['credit_score'] > 650).astype(int) * 2 +
        
        # Experience factor
        (df['experience_years'] > 5).astype(int) * 1 +
        
        # Home ownership factor
        df['owns_home'].astype(int) * 1 +
        
        # Job stability (certain job categories)
        df['job_category'].isin(['Technology', 'Finance', 'Healthcare', 'Government']).astype(int) * 1 +
        
        # Debt ratio factor (lower is better)
        (df['debt_ratio'] < 0.3).astype(int) * 1 +
        
        # Random noise to make it realistic
        np.random.choice([0, 1, -1], n_samples, p=[0.7, 0.2, 0.1])
    )
    
    # Convert to binary classification (approved/rejected)
    # Use a threshold that creates a reasonable class distribution
    threshold = approval_score.quantile(0.65)  # Approve top 35%
    df['loan_approved'] = (approval_score >= threshold).astype(int)
    
    return df

def test_realistic_training():
    """Test realistic model training with comprehensive parameter grids"""
    
    print("🚀 TESTING REALISTIC ML IMPLEMENTATION WITH COMPREHENSIVE PARAMETER GRIDS")
    print("="*100)
    
    # Create comprehensive test dataset
    df = create_comprehensive_dataset(2000)  # Larger, more complex dataset
    test_file = "comprehensive_test_data.csv"
    df.to_csv(test_file, index=False)
    
    print(f"📊 Created comprehensive test dataset: {df.shape}")
    print(f"   Target distribution: {df['loan_approved'].value_counts().to_dict()}")
    print(f"   Features: {list(df.columns[:-1])}")
    print(f"   Numeric features: {len(df.select_dtypes(include=[np.number]).columns) - 1}")  # -1 for target
    print(f"   Categorical features: {len(df.select_dtypes(include=['object', 'bool']).columns) - 1}")
    
    # Initialize realistic ML core
    ml_core = RealisticMLCore()
    
    # Test models with truly comprehensive parameter grids
    test_models = [
        "Random Forest",
        "Neural Network",
        "Support Vector Machine",
    ]
    
    all_results = []
    
    for i, model_name in enumerate(test_models):
        print(f"\n{'='*100}")
        print(f"🤖 TESTING MODEL {i+1}/{len(test_models)}: {model_name.upper()}")
        print(f"{'='*100}")
        
        result = ml_core.train_specific_model(
            file_path=test_file,
            model_name=model_name,
            target_column='loan_approved'
        )
        
        all_results.append(result)
        
        if result['success']:
            print(f"\n✅ SUCCESS: {model_name}")
            print(f"   Training time: {result['training_time']}")
            print(f"   {result['score_name']}: {result['main_score']:.4f}")
            print(f"   Complexity: {result['complexity_level']}")
            print(f"   Configurations tested: {result['parameter_configurations_tested']:,}")
            print(f"   Total model fits: {result['total_model_fits']:,}")
        else:
            print(f"\n❌ FAILED: {model_name}")
            print(f"   Error: {result['error']}")
    
    # Comprehensive analysis
    print(f"\n{'='*100}")
    print("📋 COMPREHENSIVE ANALYSIS")
    print(f"{'='*100}")
    
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    print(f"\n✅ Successful trainings: {len(successful)}/{len(all_results)}")
    
    if successful:
        print(f"\n🏆 TRAINING RESULTS SUMMARY:")
        print(f"{'Model':<20} {'Time':<15} {'Score':<10} {'Complexity':<12} {'Configs':<10} {'Total Fits':<12}")
        print(f"{'-'*85}")
        
        total_training_time = 0
        for result in successful:
            time_str = result['training_time']
            time_val = float(time_str.replace(' seconds', ''))
            total_training_time += time_val
            
            print(f"{result['model_name']:<20} {time_str:<15} {result['main_score']:<10.4f} "
                  f"{result['complexity_level']:<12} {result['parameter_configurations_tested']:<10,} "
                  f"{result['total_model_fits']:<12,}")
        
        print(f"\n📊 TRAINING TIME ANALYSIS:")
        training_times = []
        for result in successful:
            time_str = result['training_time']
            time_val = float(time_str.replace(' seconds', ''))
            training_times.append(time_val)
            
            # Realistic time assessment
            if time_val >= 30:
                status = "🏆 EXCELLENT (Very Realistic)"
            elif time_val >= 15:
                status = "✅ GOOD (Realistic)"
            elif time_val >= 5:
                status = "⚠️  FAIR (Somewhat Realistic)"
            else:
                status = "❌ POOR (Too Fast)"
            
            print(f"   {result['model_name']}: {time_val:.1f}s {status}")
        
        if training_times:
            avg_time = sum(training_times) / len(training_times)
            min_time = min(training_times)
            max_time = max(training_times)
            
            print(f"\n📈 OVERALL TRAINING STATISTICS:")
            print(f"   Average training time: {avg_time:.1f} seconds ({avg_time/60:.1f} minutes)")
            print(f"   Range: {min_time:.1f}s - {max_time:.1f}s")
            print(f"   Total training time: {total_training_time:.1f} seconds ({total_training_time/60:.1f} minutes)")
            
            # Overall assessment
            if avg_time >= 20:
                print(f"   🎉 EXCELLENT: Training times are very realistic!")
                print(f"   ✅ Successfully achieved the goal of realistic model training")
            elif avg_time >= 10:
                print(f"   ✅ GOOD: Training times are realistic")
                print(f"   📈 Significant improvement from the original 0.13 seconds")
            elif avg_time >= 5:
                print(f"   ⚠️  FAIR: Training times are somewhat realistic")
                print(f"   📊 Better than before, but could be improved further")
            else:
                print(f"   ❌ NEEDS IMPROVEMENT: Training times still too fast")
        
        # Parameter space analysis
        print(f"\n🔍 PARAMETER SPACE ANALYSIS:")
        total_configs = sum(r['parameter_configurations_tested'] for r in successful)
        total_fits = sum(r['total_model_fits'] for r in successful)
        
        print(f"   Total parameter configurations tested: {total_configs:,}")
        print(f"   Total model fits performed: {total_fits:,}")
        print(f"   Average configurations per model: {total_configs // len(successful):,}")
        print(f"   Average fits per model: {total_fits // len(successful):,}")
        
        # Model complexity analysis
        print(f"\n⚙️  MODEL COMPLEXITY ANALYSIS:")
        complexity_counts = {}
        for result in successful:
            complexity = result['complexity_level']
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        for complexity, count in complexity_counts.items():
            print(f"   {complexity}: {count} model(s)")
    
    if failed:
        print(f"\n💥 FAILED MODELS:")
        for result in failed:
            print(f"   {result['model_name']}: {result['error']}")
    
    # Cleanup
    try:
        os.remove(test_file)
        print(f"\n🧹 Cleaned up test file")
    except:
        pass

    return all_results

if __name__ == "__main__":
    results = test_realistic_training()