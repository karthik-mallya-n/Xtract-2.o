#!/usr/bin/env python3
"""
Test script to verify training details are properly displayed in the new structure.
This tests the fix for training details showing "N/A" in the UI.
"""
import requests
import json
import time

def test_training_details():
    """Test that training details are properly returned for UI display"""
    print("🧪 Testing Training Details Structure...")
    
    # Training request payload
    training_payload = {
        'file_id': 'PQCAMTFM5P',  # Using the housing dataset
        'model_name': 'Random Forest',
        'is_labeled': 'true',  # Ensure we get classification
        'test_split': 0.2
    }
    
    print(f"📤 Sending training request: {json.dumps(training_payload, indent=2)}")
    
    try:
        # Send training request
        response = requests.post(
            'http://localhost:5000/api/train-specific-model',
            json=training_payload,
            headers={'Content-Type': 'application/json'},
            timeout=300  # 5 minutes for training
        )
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Training completed successfully!")
            
            # Check if training_details exists at top level
            if 'training_details' in result:
                training_details = result['training_details']
                print(f"✅ Found training_details at top level")
                print(f"📊 Training details structure:")
                print(json.dumps(training_details, indent=2))
                
                # Check for specific metrics
                metrics_to_check = ['accuracy', 'precision', 'recall', 'f1_score']
                for metric in metrics_to_check:
                    if metric in training_details:
                        value = training_details[metric]
                        if value and value != 'N/A' and value != 'NaN':
                            print(f"✅ {metric}: {value}")
                        else:
                            print(f"❌ {metric}: {value} (showing as N/A)")
                    else:
                        print(f"⚠️  {metric}: Not found")
                        
            else:
                print(f"❌ training_details not found at top level")
                print(f"🔍 Available keys: {list(result.keys())}")
                
            # Also check the full response structure
            print(f"\n🔍 Full response structure:")
            for key in result.keys():
                print(f"  - {key}: {type(result[key])}")
                
        else:
            print(f"❌ Training failed with status {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ Failed to decode JSON response: {e}")

def test_prediction_with_debug():
    """Test prediction with enhanced debugging"""
    print("\n🧪 Testing Prediction with Debug Info...")
    
    # Simple prediction payload
    prediction_payload = {
        'features': [1500, 3, 2, 1, 0.5, 2000, 1990, 98115, 47.68, -122.28, 1690, 8062]
    }
    
    print(f"📤 Sending prediction request: {json.dumps(prediction_payload, indent=2)}")
    
    try:
        response = requests.post(
            'http://localhost:5000/api/predict',
            json=prediction_payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        print(f"📥 Prediction response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            print(f"🔮 Prediction result: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Prediction failed with status {response.status_code}")
            error_data = response.json() if response.content else {}
            print(f"Error details: {json.dumps(error_data, indent=2)}")
            
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Training and Prediction Tests...\n")
    
    # Test training details
    test_training_details()
    
    # Wait a moment then test prediction
    time.sleep(2)
    test_prediction_with_debug()
    
    print("\n✅ Tests completed!")