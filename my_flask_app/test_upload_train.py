#!/usr/bin/env python3
"""
Test script to upload a file and immediately test training.
"""
import requests
import json
import time

def upload_test_file():
    """Upload a test CSV file"""
    print("📤 Uploading test file...")
    
    # Create a larger test dataset with at least 20 samples for proper train/test split
    test_data = """Store,Date,Weekly_Sales,Holiday_Flag,Temperature,Fuel_Price,CPI,Unemployment
1,2010-02-05,1643690.90,0,42.31,2.572,211.0963582,8.106
1,2010-02-12,1641957.44,1,38.51,2.548,211.2420151,8.106
1,2010-02-19,1611968.17,0,39.93,2.514,211.2891429,8.106
1,2010-02-26,1409727.59,0,46.63,2.561,211.3196429,8.106
1,2010-03-05,1554806.68,0,46.50,2.625,211.3501429,8.106
1,2010-03-12,1621348.97,0,57.79,2.667,211.3806429,8.106
1,2010-03-19,1545552.86,0,54.58,2.720,211.4011429,8.106
1,2010-03-26,1309631.51,0,51.45,2.732,211.5811429,8.106
1,2010-04-02,1370978.00,0,56.47,2.760,211.9801429,8.106
1,2010-04-09,1528808.79,0,58.85,2.778,212.0426429,8.106
2,2010-02-05,2343690.90,0,45.31,2.572,211.0963582,7.806
2,2010-02-12,2441957.44,1,41.51,2.548,211.2420151,7.806
2,2010-02-19,2311968.17,0,42.93,2.514,211.2891429,7.806
2,2010-02-26,2209727.59,0,49.63,2.561,211.3196429,7.806
2,2010-03-05,2254806.68,0,49.50,2.625,211.3501429,7.806
2,2010-03-12,2321348.97,0,60.79,2.667,211.3806429,7.806
2,2010-03-19,2245552.86,0,57.58,2.720,211.4011429,7.806
2,2010-03-26,2109631.51,0,54.45,2.732,211.5811429,7.806
2,2010-04-02,2170978.00,0,59.47,2.760,211.9801429,7.806
2,2010-04-09,2228808.79,0,61.85,2.778,212.0426429,7.806
3,2010-02-05,1143690.90,0,40.31,2.572,211.0963582,9.206
3,2010-02-12,1241957.44,1,36.51,2.548,211.2420151,9.206
3,2010-02-19,1111968.17,0,37.93,2.514,211.2891429,9.206
3,2010-02-26,1009727.59,0,44.63,2.561,211.3196429,9.206
3,2010-03-05,1054806.68,0,44.50,2.625,211.3501429,9.206"""
    
    files = {'file': ('test_sales.csv', test_data, 'text/csv')}
    
    # Include required form data
    data = {
        'is_labeled': 'labeled',
        'data_type': 'continuous'
    }
    
    try:
        response = requests.post(
            'http://localhost:5000/api/upload',
            files=files,
            data=data,
            timeout=30
        )
        
        print(f"📥 Upload response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ File uploaded successfully!")
            print(f"📄 File ID: {result['file_id']}")
            return result['file_id']
        else:
            print(f"❌ Upload failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None

def test_immediate_training(file_id):
    """Test training immediately after upload"""
    print(f"\n🧪 Testing Training with File ID: {file_id}")
    
    training_payload = {
        'file_id': file_id,
        'model_name': 'Random Forest',
        'target_column': 'Unemployment',  # Use the target column from our test data
        'is_labeled': 'true',
        'test_split': 0.2
    }
    
    print(f"📤 Sending training request: {json.dumps(training_payload, indent=2)}")
    
    try:
        response = requests.post(
            'http://localhost:5000/api/train-specific-model',
            json=training_payload,
            headers={'Content-Type': 'application/json'},
            timeout=300
        )
        
        print(f"📥 Training response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Training completed successfully!")
            
            # Check for training_details at top level
            if 'training_details' in result:
                training_details = result['training_details']
                print(f"\n✅ Training Details Found at Top Level!")
                print(f"📊 Training Details:")
                print(json.dumps(training_details, indent=2))
                
                # Check performance separately
                if 'performance' in result:
                    performance = result['performance']
                    print(f"\n📈 Performance Metrics Found Separately!")
                    print(f"📊 Performance:")
                    print(json.dumps(performance, indent=2))
                
                # Verify specific metrics in both locations
                metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'r2_score']
                print(f"\n🔍 Metrics Verification:")
                for metric in metrics:
                    in_training_details = metric in training_details
                    in_performance = 'performance' in result and metric in result['performance']
                    
                    if in_training_details:
                        value = training_details[metric]
                        print(f"  ✅ {metric} (in training_details): {value}")
                    elif in_performance:
                        value = result['performance'][metric]
                        print(f"  ⚠️  {metric} (in performance): {value}")
                    else:
                        print(f"  ❌ {metric}: Not found anywhere")
            else:
                print(f"❌ training_details not found at top level!")
                print(f"🔍 Available keys: {list(result.keys())}")
                
                # Also check if performance exists separately
                if 'performance' in result:
                    print(f"📈 But performance data found separately:")
                    print(json.dumps(result['performance'], indent=2))
                
        else:
            error_data = response.json() if response.content else {}
            print(f"❌ Training failed: {json.dumps(error_data, indent=2)}")
            
    except Exception as e:
        print(f"❌ Training test failed: {e}")

if __name__ == "__main__":
    print("🚀 Testing Upload + Training Flow...\n")
    
    # Upload file first
    file_id = upload_test_file()
    
    if file_id:
        # Test training immediately
        test_immediate_training(file_id)
    else:
        print("❌ Cannot proceed without successful file upload")
    
    print("\n✅ Test completed!")