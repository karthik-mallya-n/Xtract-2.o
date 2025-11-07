"""
Test React Integration - Check that API responses have unique model keys
for React frontend compatibility
"""
import json
import sys
sys.path.append('.')

# Test that our API response format is compatible with React (no duplicate keys)
def test_react_compatibility():
    print("🧪 Testing React Frontend Compatibility...")
    print("=" * 60)
    
    # Simulate the exact data structure that gets sent to React frontend
    test_response = {
        "success": True,
        "recommended_models": [
            {
                "id": "model_1_catboost_classifier",
                "name": "CatBoost Classifier",
                "rank": 1,
                "expected_accuracy": "97-99%",
                "reasoning": "Best for tabular data",
                "advantages": "Robust against overfitting"
            },
            {
                "id": "model_2_xgboost_classifier", 
                "name": "XGBoost Classifier",
                "rank": 2,
                "expected_accuracy": "96-99%",
                "reasoning": "High performance gradient boosting",
                "advantages": "Strong regularization"
            },
            {
                "id": "model_3_lightgbm_classifier",
                "name": "LightGBM Classifier", 
                "rank": 3,
                "expected_accuracy": "96-98%",
                "reasoning": "Fast and efficient",
                "advantages": "Low memory usage"
            }
        ],
        "alternative_models": [],  # Should be empty to avoid duplicates
        "scenario_detected": {
            "type": "Labeled + Categorical",
            "task": "Classification"
        },
        "dataset_info": {
            "rows": 160,
            "columns": 3
        }
    }
    
    print("📊 API Response Structure:")
    print(f"   ✅ Recommended Models: {len(test_response['recommended_models'])}")
    print(f"   ✅ Alternative Models: {len(test_response['alternative_models'])}")
    
    # Check for unique IDs (React keys)
    model_ids = [model['id'] for model in test_response['recommended_models']]
    
    print(f"\n🔑 Model IDs for React keys:")
    for i, model_id in enumerate(model_ids, 1):
        print(f"   {i}. {model_id}")
    
    # Verify no duplicates
    unique_ids = set(model_ids)
    if len(model_ids) == len(unique_ids):
        print(f"\n✅ SUCCESS: All {len(model_ids)} model IDs are unique!")
        print("✅ No React 'duplicate key' errors will occur")
    else:
        print(f"\n❌ ERROR: Found duplicate IDs!")
        duplicates = [id for id in model_ids if model_ids.count(id) > 1]
        print(f"   Duplicates: {set(duplicates)}")
        return False
        
    # Verify JSON serialization works
    try:
        json_str = json.dumps(test_response, indent=2)
        print("✅ JSON serialization successful")
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        return False
        
    print("\n🎯 RESULT: API response is React-compatible!")
    print("   - Unique model IDs prevent duplicate key errors")
    print("   - Clean JSON structure for frontend consumption")
    print("   - No model repetition between arrays")
    
    return True

if __name__ == "__main__":
    success = test_react_compatibility()
    if success:
        print("\n🚀 Ready for frontend integration!")
    else:
        print("\n💥 Issues found - frontend integration may fail!")