import pandas as pd
import time
from core_ml import MLCore

# Create test data
data = {
    'credit_score': [750, 680, 620, 580, 720, 600, 650, 590, 710, 630,
                     780, 690, 640, 570, 730, 610, 660, 580, 720, 650,
                     760, 700, 630, 560, 740, 620, 670, 590, 730, 640,
                     770, 710, 650, 550, 750, 630, 680, 600, 740, 660],
    'income': [60000, 45000, 35000, 30000, 55000, 32000, 42000, 28000, 58000, 38000,
               62000, 47000, 36000, 31000, 56000, 33000, 43000, 29000, 59000, 39000,
               64000, 48000, 37000, 32000, 57000, 34000, 44000, 30000, 60000, 40000,
               66000, 49000, 38000, 33000, 58000, 35000, 45000, 31000, 61000, 41000],
    'age': [35, 28, 45, 52, 31, 38, 42, 29, 36, 48,
            37, 30, 47, 54, 33, 40, 44, 31, 38, 50,
            39, 32, 49, 56, 35, 42, 46, 33, 40, 52,
            41, 34, 51, 58, 37, 44, 48, 35, 42, 54],
    'loan_approved': ['Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No',
                      'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No',
                      'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No',
                      'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)
df.to_csv('speed_test_data.csv', index=False)

print("🚀 SPEED TEST: Optimized Training System")
print("=" * 60)

# Initialize ML Core
import os
os.environ['GEMINI_API_KEY'] = 'fake_api_key'  # Set environment variable
ml_core = MLCore()  # Initialize without parameters

# Test optimized Random Forest training
start_time = time.time()
print(f"\n⏱️  Testing optimized Random Forest training...")
print(f"🎯 Target: Under 60 seconds (vs 10+ minutes before)")

result = ml_core.train_advanced_model(
    model_name='Random Forest',
    file_path='speed_test_data.csv',
    target_column='loan_approved'
)

end_time = time.time()
training_time = end_time - start_time

print(f"\n📊 SPEED TEST RESULTS")
print("=" * 40)
print(f"⏱️  Training Time: {training_time:.1f} seconds")
print(f"🎯 Target Time: 60 seconds")
print(f"✅ Speed Improvement: {(600/training_time):.1f}x faster" if training_time < 60 else "❌ Still too slow")

if result['success']:
    print(f"🎯 Accuracy: {result['main_score']:.3f} ({result['main_score']*100:.1f}%)")
    print(f"✅ Success: {result['threshold_met']}")
else:
    print(f"❌ Training failed: {result.get('error', 'Unknown error')}")

# Clean up
import os
if os.path.exists('speed_test_data.csv'):
    os.remove('speed_test_data.csv')