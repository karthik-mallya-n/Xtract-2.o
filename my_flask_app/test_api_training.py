import requests
import pandas as pd
import json
import time

# Create a simple test dataset
data = {
    'credit_score': [750, 680, 620, 580, 720, 600, 650, 590, 710, 630,
                     780, 690, 640, 570, 730, 610, 660, 580, 720, 650,
                     760, 700, 630, 560, 740, 620, 670, 590, 730, 640],
    'income': [60000, 45000, 35000, 30000, 55000, 32000, 42000, 28000, 58000, 38000,
               62000, 47000, 36000, 31000, 56000, 33000, 43000, 29000, 59000, 39000,
               64000, 48000, 37000, 32000, 57000, 34000, 44000, 30000, 60000, 40000],
    'age': [35, 28, 45, 52, 31, 38, 42, 29, 36, 48,
            37, 30, 47, 54, 33, 40, 44, 31, 38, 50,
            39, 32, 49, 56, 35, 42, 46, 33, 40, 52],
    'loan_approved': ['Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No',
                      'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No',
                      'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)
df.to_csv('api_test_data.csv', index=False)

def test_training_api():
    url = 'http://localhost:5000/api/train-advanced'
    
    # Test data
    payload = {
        'model_name': 'Random Forest',
        'dataset_filename': 'api_test_data.csv',
        'target_column': 'loan_approved'
    }
    
    print("🧪 Testing Flask Training API")
    print("=" * 50)
    print(f"📄 Dataset: {payload['dataset_filename']}")
    print(f"🤖 Model: {payload['model_name']}")
    print(f"🎯 Target: {payload['target_column']}")
    print()
    
    try:
        # Upload dataset first
        with open('api_test_data.csv', 'rb') as f:
            files = {'file': f}
            data = {
                'is_labeled': 'labeled',
                'data_type': 'continuous'
            }
            upload_response = requests.post('http://localhost:5000/api/upload', files=files, data=data)
            print(f"📤 Upload Status: {upload_response.status_code}")
            if upload_response.status_code == 200:
                print("✅ Dataset uploaded successfully")
                upload_data = upload_response.json()
                payload['file_id'] = upload_data.get('file_id')  # Add file_id from upload
            else:
                print(f"❌ Upload failed: {upload_response.text}")
                return
        
        # Start training
        print("\n🚀 Starting training...")
        response = requests.post(url, json=payload, timeout=120)
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Training completed successfully!")
            print(f"📊 Full Response: {json.dumps(result, indent=2)}")
            print(f"📊 Accuracy: {result.get('accuracy', 'N/A')}")
            print(f"📁 Model Path: {result.get('model_path', 'N/A')}")
            print(f"🔧 Best Params: {result.get('best_params', 'N/A')}")
        else:
            print(f"❌ Training failed: {response.text}")
            print(f"📊 Status Code: {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
    except requests.exceptions.ConnectionError:
        print("🔌 Connection failed - is Flask server running?")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        # Clean up
        import os
        if os.path.exists('api_test_data.csv'):
            os.remove('api_test_data.csv')
            print("\n🧹 Cleaned up test file")

if __name__ == "__main__":
    test_training_api()