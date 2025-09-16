"""
Test script for ML Platform Backend API
This script tests all the main API endpoints to ensure they're working correctly.
"""

import requests
import json
import time
import os

# API base URL
BASE_URL = "http://localhost:5000/api"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to API: {e}")
        return False

def test_upload_endpoint():
    """Test file upload endpoint with a sample CSV"""
    print("\n📤 Testing file upload endpoint...")
    
    # Create a sample CSV file for testing
    sample_csv_content = """Name,Age,Salary,Department
John Doe,25,50000,Engineering
Jane Smith,30,60000,Marketing
Bob Johnson,35,70000,Sales
Alice Brown,28,55000,Engineering
Charlie Davis,32,65000,Marketing"""
    
    # Write sample file
    test_file_path = "test_sample.csv"
    with open(test_file_path, 'w') as f:
        f.write(sample_csv_content)
    
    try:
        # Prepare the file upload
        with open(test_file_path, 'rb') as f:
            files = {'file': ('test_sample.csv', f, 'text/csv')}
            data = {
                'is_labeled': 'labeled',
                'data_type': 'categorical'
            }
            
            response = requests.post(f"{BASE_URL}/upload", files=files, data=data, timeout=30)
            
        if response.status_code == 200:
            result = response.json()
            print(f"✅ File upload successful: {result['filename']}")
            print(f"   File ID: {result['file_id']}")
            print(f"   File size: {result['file_size']} bytes")
            
            # Clean up test file
            os.remove(test_file_path)
            
            return result['file_id']
        else:
            print(f"❌ File upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            # Clean up test file
            if os.path.exists(test_file_path):
                os.remove(test_file_path)
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Upload request failed: {e}")
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
        return None

def test_recommend_model_endpoint(file_id):
    """Test model recommendation endpoint"""
    print(f"\n🤖 Testing model recommendation endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/recommend-model", 
                              params={'file_id': file_id}, 
                              timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Model recommendation successful")
            print(f"   Dataset rows: {result['dataset_info']['total_rows']}")
            print(f"   Dataset columns: {result['dataset_info']['total_columns']}")
            
            # Check if LLM response was successful
            if 'recommendations' in result:
                print("   LLM recommendations received")
            else:
                print("   ⚠️ LLM recommendations not available (OpenRouter API key needed)")
            
            return True
        else:
            print(f"❌ Model recommendation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Recommendation request failed: {e}")
        return False

def test_models_endpoint():
    """Test list models endpoint"""
    print(f"\n📋 Testing list models endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Models list retrieved: {result['count']} models found")
            return True
        else:
            print(f"❌ Models list failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Models request failed: {e}")
        return False

def main():
    """Run all API tests"""
    print("🚀 ML Platform Backend API Test Suite")
    print("=" * 50)
    
    # Test health endpoint first
    if not test_health_endpoint():
        print("\n❌ Backend is not running or not accessible.")
        print("Please start the backend with: python app.py")
        return
    
    # Test file upload
    file_id = test_upload_endpoint()
    if not file_id:
        print("\n❌ File upload failed. Cannot continue with other tests.")
        return
    
    # Test model recommendation
    test_recommend_model_endpoint(file_id)
    
    # Test list models
    test_models_endpoint()
    
    print("\n" + "=" * 50)
    print("🏁 API tests completed!")
    print("\n💡 Tips:")
    print("   - For full functionality, set OPENROUTER_API_KEY in .env file")
    print("   - Start Celery worker for training: celery -A tasks worker --loglevel=info")
    print("   - Training and prediction endpoints require Celery worker")

if __name__ == "__main__":
    main()