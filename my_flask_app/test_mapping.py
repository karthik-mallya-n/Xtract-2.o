import pandas as pd
import time
from core_ml import MLCore
import os

# Set environment variable
os.environ['GEMINI_API_KEY'] = 'fake_api_key'

# Create Iris dataset for testing
iris_data = {
    'Id': range(1, 151),
    'SepalLengthCm': [5.1, 4.9, 4.7, 4.6, 5.0] * 30,
    'SepalWidthCm': [3.5, 3.0, 3.2, 3.1, 3.6] * 30,
    'PetalLengthCm': [1.4, 1.4, 1.3, 1.5, 1.4] * 30,
    'PetalWidthCm': [0.2, 0.2, 0.2, 0.2, 0.2] * 30,
    'Species': ['Iris-setosa'] * 50 + ['Iris-versicolor'] * 50 + ['Iris-virginica'] * 50
}

df = pd.DataFrame(iris_data)
df.to_csv('test_iris.csv', index=False)

print("🧪 TESTING FRONTEND MODEL NAME MAPPING")
print("=" * 60)

# Initialize ML Core
ml_core = MLCore()

# Test the frontend model name 'random-forest-classifier'
start_time = time.time()
print(f"\n⏱️  Testing 'random-forest-classifier' mapping...")

result = ml_core.train_advanced_model(
    model_name='random-forest-classifier',
    file_path='test_iris.csv',
    target_column='Species'
)

end_time = time.time()
training_time = end_time - start_time

print(f"\n📊 MAPPING TEST RESULTS")
print("=" * 40)
print(f"⏱️  Training Time: {training_time:.1f} seconds")
print(f"✅ Mapping Success: {'Yes' if result['success'] else 'No'}")

if result['success']:
    print(f"🎯 Accuracy: {result['main_score']:.3f} ({result['main_score']*100:.1f}%)")
    print(f"📁 Model Saved: {result['model_folder']}")
    print(f"✅ Threshold Met: {result['threshold_met']}")
else:
    print(f"❌ Error: {result.get('error', 'Unknown error')}")

# Clean up
if os.path.exists('test_iris.csv'):
    os.remove('test_iris.csv')
    print(f"\n🧹 Cleaned up test file")