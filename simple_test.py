#!/usr/bin/env python3
"""
Simple test to just test training with existing file
"""

import requests
import json

# Test configuration
FLASK_URL = "http://localhost:5000"
TEST_MODEL = "Lasso Regression"
FILE_ID = "e825f3a3-8fd7-4736-a382-041b945c946c"  # From previous test

def test_training():
    """Test training with existing file"""
    print("🚀 Testing training...")
    
    train_data = {
        'file_id': FILE_ID,
        'model_name': TEST_MODEL
    }
    
    print(f"Request: {train_data}")
    
    response = requests.post(f"{FLASK_URL}/api/train", 
                           json=train_data, 
                           timeout=300)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Success!")
        print(f"Full result:")
        print(json.dumps(result, indent=2))
    else:
        print(f"❌ Failed: {response.text}")

if __name__ == "__main__":
    test_training()