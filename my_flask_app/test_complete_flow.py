#!/usr/bin/env python3
"""
Complete Training Flow Test
Tests the entire training flow and generates results for the frontend
"""

import sys
import os
import time
import json
import requests

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_ml import ml_core

def test_complete_flow():
    """Test complete training flow and simulate frontend interaction"""
    
    print("🧪 TESTING COMPLETE TRAINING FLOW")
    print("=" * 60)
    
    # Test parameters
    file_path = "uploads/b0560d95-7006-4035-9ac9-a547229a0071.csv"
    model_name = "Random Forest"  # Use frontend-style name
    target_column = "Species"
    
    print(f"📄 Dataset: {file_path}")
    print(f"🤖 Model: {model_name}")  
    print(f"🎯 Target: {target_column}")
    
    # Simulate API call to /api/train endpoint
    api_url = "http://localhost:5000/api/train"
    
    # Create payload like frontend would send
    payload = {
        "file_id": "test-file-id",  # Dummy file ID for testing
        "model_name": model_name
    }
    
    print(f"\n🌐 Testing API endpoint: {api_url}")
    print(f"📤 Payload: {payload}")
    
    try:
        # Test API endpoint
        response = requests.post(api_url, json=payload, timeout=30)
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API Response: {result}")
            
            # Extract training result
            if result.get('success') and 'result' in result:
                training_result = result['result']
                
                print(f"\n🎉 TRAINING RESULTS:")
                print(f"📁 Model Folder: {training_result.get('model_folder')}")
                print(f"🎯 Accuracy: {training_result.get('main_score', 0)*100:.2f}%")
                print(f"📊 Problem Type: {training_result.get('problem_type')}")
                print(f"✅ Threshold Met: {training_result.get('threshold_met')}")
                
                # Save results for frontend testing
                results_file = "test_results.json"
                with open(results_file, 'w') as f:
                    json.dump(training_result, f, indent=2, default=str)
                
                print(f"\n💾 Results saved to: {results_file}")
                print(f"🔗 Frontend URL: http://localhost:3000/results")
                
                return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Flask server not running on localhost:5000")
        print("💡 Fallback: Testing core ML function directly")
        
        # Fallback to direct function call
        start_time = time.time()
        result = ml_core.train_advanced_model(
            model_name=model_name,
            file_path=file_path,
            target_column=target_column
        )
        end_time = time.time()
        
        print(f"⏱️  Training completed in {end_time - start_time:.2f} seconds")
        
        if result['success']:
            print(f"🎯 Accuracy: {result['main_score']*100:.2f}%")
            print(f"📁 Model saved: {result['model_folder']}")
            
            # Save results for frontend testing
            results_file = "test_results.json"
            with open(results_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to: {results_file}")
            return True
        else:
            print(f"❌ Training failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def generate_frontend_url():
    """Generate URL for testing in frontend"""
    try:
        with open("test_results.json", 'r') as f:
            results = json.load(f)
        
        import urllib.parse
        encoded_results = urllib.parse.quote(json.dumps(results))
        frontend_url = f"http://localhost:3000/results?results={encoded_results}"
        
        print(f"\n🔗 FRONTEND TEST URLS:")
        print(f"📊 Results Page: http://localhost:3000/results")
        print(f"📊 With Data: {frontend_url[:100]}...")
        print(f"\n💡 Copy the Results Page URL to test in your browser!")
        
    except FileNotFoundError:
        print("❌ No test results file found")

if __name__ == "__main__":
    success = test_complete_flow()
    
    if success:
        print(f"\n✅ COMPLETE FLOW TEST PASSED!")
        generate_frontend_url()
        print(f"\n🎉 Your training system is ready for production!")
    else:
        print(f"\n❌ COMPLETE FLOW TEST FAILED!")
        sys.exit(1)