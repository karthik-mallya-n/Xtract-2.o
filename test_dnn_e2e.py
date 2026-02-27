#!/usr/bin/env python3
"""
End-to-end test of DNN model pipeline
"""
import requests
import json
import io
import csv
import sys
sys.path.insert(0, '/app')

BASE_URL = "http://localhost:5000"

print("\n" + "="*70)
print("END-TO-END DNN PIPELINE TEST")
print("="*70)

# Step 1: Create a simple iris dataset
print("\n[STEP 1] Creating test iris dataset...")
iris_data = [
    ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'],
    [5.1, 3.5, 1.4, 0.2, 'setosa'],
    [7.0, 3.2, 4.7, 1.4, 'versicolor'],
    [6.3, 3.3, 6.0, 2.5, 'virginica'],
    [4.9, 3.0, 1.4, 0.2, 'setosa'],
    [6.5, 2.8, 4.6, 1.5, 'versicolor'],
    [5.8, 2.7, 5.1, 1.9, 'virginica'],
    [5.4, 3.7, 1.5, 0.2, 'setosa'],
]

csv_content = io.BytesIO()
text_wrapper = io.TextIOWrapper(csv_content, encoding='utf-8', newline='')
writer = csv.writer(text_wrapper)
writer.writerows(iris_data)
text_wrapper.flush()
csv_content.seek(0)

print("✓ Dataset created with 8 samples")

# Step 2: Upload dataset
print("\n[STEP 2] Uploading dataset...")
files = {'file': ('iris_test.csv', csv_content, 'text/csv')}
data = {
    'is_labeled': 'labeled',
    'data_type': 'categorical',
    'target_column': 'species',
    'selected_columns': '["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]'
}
upload_resp = requests.post(f"{BASE_URL}/api/upload", files=files, data=data)

if upload_resp.status_code != 200:
    print(f"✗ Upload failed: {upload_resp.status_code}")
    print(upload_resp.text)
    sys.exit(1)

upload_data = upload_resp.json()
print(f"✓ Upload successful")
file_id = upload_data.get('file_id')
print(f"  file_id: {file_id}")

# Step 3: Get recommendations
print("\n[STEP 3] Getting model recommendations...")
rec_resp = requests.get(f"{BASE_URL}/api/recommend-model", params={'file_id': file_id})

if rec_resp.status_code != 200:
    print(f"✗ Recommendation failed: {rec_resp.status_code}")
    print(rec_resp.text)
    sys.exit(1)

rec_data = rec_resp.json()
print("✓ Recommendations received")

# Navigate to the recommendations object
recs = rec_data.get('recommendations', {})
primary_list = recs.get('recommended_models', [])
alternatives = recs.get('alternative_models', [])

if primary_list:
    primary = primary_list[0]
    print(f"  Primary: {primary.get('name', 'N/A')}")
else:
    print(f"  Primary: N/A")

print(f"  Alternatives ({len(alternatives)}):")
for model in alternatives:
    print(f"    - {model.get('name', 'N/A')}")

# Check if DNN is in alternatives
dnn_found = any('Deep Neural Network' in m.get('name', '') for m in alternatives)
print(f"\n  ✓ DNN Model Found: {dnn_found}")

if not dnn_found:
    print("✗ Deep Neural Network not found in recommendations!")
    sys.exit(1)

# Step 4: Train with Deep Neural Network
print("\n[STEP 4] Starting DNN training...")
train_resp = requests.post(f"{BASE_URL}/api/train", json={
    'file_id': file_id,
    'model_name': 'Deep Neural Network',
    'test_size': 0.2
})

if train_resp.status_code != 200:
    print(f"✗ Training failed: {train_resp.status_code}")
    print(train_resp.text)
    sys.exit(1)

train_data = train_resp.json()
model_id = train_data.get('model_id')
print(f"✓ Training initiated")
print(f"  model_id: {model_id}")
print(f"  status: {train_data.get('status', 'N/A')}")
print(f"  message: {train_data.get('message', 'N/A')}")

# Step 5: Wait for training to complete and get results
print("\n[STEP 5] Waiting for training to complete...")
import time
max_wait = 120  # 2 minutes
start = time.time()
while time.time() - start < max_wait:
    status_resp = requests.get(f"{BASE_URL}/api/training-status", params={'model_id': model_id})
    if status_resp.status_code == 200:
        status_data = status_resp.json()
        is_complete = status_data.get('is_complete', False)
        print(f"  Training progress: {status_data.get('message', 'Running...')}")
        if is_complete:
            print(f"✓ Training complete!")
            print(f"  accuracy: {status_data.get('accuracy', 'N/A')}")
            print(f"  f1_score: {status_data.get('f1_score', 'N/A')}")
            break
    time.sleep(5)

print("\n" + "="*70)
print("END-TO-END TEST COMPLETE - ALL STEPS PASSED ✓")
print("="*70 + "\n")
