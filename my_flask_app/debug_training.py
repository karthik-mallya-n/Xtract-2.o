#!/usr/bin/env python3
"""
Debug script to see the full structure of training results.
"""
import requests
import json

def debug_training_response():
    """Debug the full training response structure"""
    print("🐛 Debugging Full Training Response Structure...")
    
    # Upload test file
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
    
    files = {'file': ('debug_dataset.csv', test_data, 'text/csv')}
    data = {'is_labeled': 'labeled', 'data_type': 'continuous'}
    
    # Upload
    upload_response = requests.post('http://localhost:5000/api/upload', files=files, data=data)
    if upload_response.status_code != 200:
        print(f"❌ Upload failed: {upload_response.text}")
        return
        
    file_id = upload_response.json()['file_id']
    print(f"✅ File uploaded with ID: {file_id}")
    
    # Train
    training_payload = {
        'file_id': file_id,
        'model_name': 'Random Forest',
        'target_column': 'Unemployment',
        'is_labeled': 'true',
        'test_split': 0.2
    }
    
    training_response = requests.post(
        'http://localhost:5000/api/train-specific-model',
        json=training_payload,
        headers={'Content-Type': 'application/json'},
        timeout=300
    )
    
    if training_response.status_code != 200:
        print(f"❌ Training failed: {training_response.text}")
        return
        
    result = training_response.json()
    print(f"✅ Training completed successfully!")
    
    # Print the FULL response structure
    print(f"\\n🔍 FULL RESPONSE STRUCTURE:")
    print(f"{'='*60}")
    print(json.dumps(result, indent=2, default=str))
    
    # Analyze structure
    print(f"\\n📋 TOP-LEVEL KEYS:")
    print(f"{'='*60}")
    for key in result.keys():
        value = result[key]
        print(f"{key}: {type(value).__name__}")
        if isinstance(value, dict):
            print(f"  └── Dict keys: {list(value.keys())}")
        elif isinstance(value, list):
            print(f"  └── List length: {len(value)}")

if __name__ == "__main__":
    debug_training_response()