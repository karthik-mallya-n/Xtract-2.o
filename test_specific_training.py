"""
Quick Test for Specific Model Training
Run this to verify the implementation works correctly
"""

import requests
import json
import time

# Configuration
BACKEND_URL = "http://localhost:5000"

def test_specific_model_training():
    """Test the /api/train endpoint with specific model training"""
    
    print("\n" + "="*100)
    print("🧪 TESTING SPECIFIC MODEL TRAINING")
    print("="*100 + "\n")
    
    # Test parameters
    test_data = {
        "file_id": "test-file-123",  # You'll need to upload a file first
        "model_name": "Random Forest"
    }
    
    print(f"📋 Test Data:")
    print(json.dumps(test_data, indent=2))
    print()
    
    print("🚀 Sending training request...")
    print(f"URL: {BACKEND_URL}/api/train")
    print()
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{BACKEND_URL}/api/train",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"⏱️  Request completed in {elapsed_time:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        print()
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ SUCCESS!")
            print(f"\n📊 Response:")
            print(json.dumps(result, indent=2))
            
            # Validate response structure
            print("\n🔍 Validating Response Structure:")
            
            checks = [
                ("success", result.get("success")),
                ("model_name", result.get("model_name")),
                ("result.success", result.get("result", {}).get("success")),
                ("result.performance", result.get("result", {}).get("performance")),
                ("result.model_info", result.get("result", {}).get("model_info")),
            ]
            
            for check_name, check_value in checks:
                status = "✅" if check_value else "❌"
                print(f"{status} {check_name}: {check_value}")
            
            # Check for comprehensive training details
            if result.get("result", {}).get("performance"):
                perf = result["result"]["performance"]
                print("\n📊 Performance Metrics:")
                for key, value in perf.items():
                    print(f"   - {key}: {value}")
            
            if result.get("result", {}).get("model_info"):
                info = result["result"]["model_info"]
                print("\n🔧 Model Info:")
                for key, value in info.items():
                    if key != "artifacts":
                        print(f"   - {key}: {value}")
            
        else:
            print(f"❌ FAILED!")
            print(f"Response: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR!")
        print("Make sure the Flask backend is running on http://localhost:5000")
        print("Run: cd my_flask_app && python app.py")
    
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*100)
    print("🧪 TEST COMPLETE")
    print("="*100 + "\n")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    SPECIFIC MODEL TRAINING TEST SCRIPT                       ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
    Before running this test:
    
    1. Start the Flask backend:
       cd my_flask_app
       python app.py
    
    2. Upload a dataset through the UI or API
    
    3. Get the file_id from the upload response
    
    4. Update the file_id in this script (line 23)
    
    5. Run this script:
       python test_specific_training.py
    
    ════════════════════════════════════════════════════════════════════════════════
    """)
    
    input("Press Enter to start the test (or Ctrl+C to cancel)...")
    test_specific_model_training()
