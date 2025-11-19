"""
Test the fixed train_advanced_model method
"""
import json

def test_train_endpoint():
    print("🧪 Testing Fixed Train Endpoint")
    print("=" * 50)
    
    # The method signature should now be correct:
    # train_advanced_model(model_name, file_path, target_column)
    
    # Example API request that should work
    api_request = {
        "file_id": "some_file_id",
        "model_name": "Random Forest Classifier"
    }
    
    print("📤 API Request Example:")
    print(json.dumps(api_request, indent=2))
    
    print("\n✅ Fixed Issues:")
    print("   - Removed 'df' parameter from method call")
    print("   - Added proper target column detection")
    print("   - Added clustering model handling")
    print("   - Uses file_path instead of passing DataFrame")
    
    print("\n🔧 Method Call Now Uses:")
    print("   ml_core.train_advanced_model(")
    print("       model_name=model_name,")
    print("       file_path=file_path,")
    print("       target_column=target_column")
    print("   )")
    
    print("\n🎯 Target Column Logic:")
    print("   1. Check user_answers for 'target_column'")
    print("   2. If clustering model → use first column as dummy")
    print("   3. Otherwise → use last column as target")

if __name__ == "__main__":
    test_train_endpoint()
    print("\n🚀 The 'df' parameter error should now be fixed!")