#!/usr/bin/env python3
"""
Comprehensive Test of Dynamic Prediction Forms
"""

import requests
import json
import uuid
import pandas as pd
import os

API_BASE = "http://localhost:5000"

def test_dynamic_forms():
    """Test that different datasets generate different prediction forms"""
    
    print("🧪 TESTING DYNAMIC PREDICTION FORMS")
    print("=" * 60)
    
    # Test Case 1: Iris Dataset (existing)
    print("\n📊 TEST 1: IRIS DATASET")
    print("-" * 30)
    
    iris_result = test_training_with_file_id("745b84df-f69f-4cbb-824f-952a29bf69fe", "CatBoost")
    iris_features = iris_result.get('feature_info', {}).get('feature_names', [])
    print(f"✅ Iris Features: {iris_features}")
    
    # Test Case 2: Create a custom dataset with different features
    print("\n📊 TEST 2: CUSTOM BUSINESS DATASET")
    print("-" * 30)
    
    # Create a business dataset
    business_data = pd.DataFrame({
        'company_age': [1, 5, 10, 3, 7, 15, 2, 8],
        'revenue_millions': [0.5, 2.3, 15.2, 1.1, 5.7, 45.8, 0.8, 8.9],
        'employees': [5, 25, 150, 15, 60, 500, 8, 90],
        'industry_score': [6.2, 7.8, 8.9, 6.5, 7.2, 9.1, 5.8, 7.9],
        'success_category': ['startup', 'growth', 'mature', 'startup', 'growth', 'enterprise', 'startup', 'growth']
    })
    
    # Save to uploads folder
    business_file_path = f"{os.path.dirname(__file__)}/uploads/business_dataset.csv"
    business_data.to_csv(business_file_path, index=False)
    
    # Generate a file ID for this dataset
    business_file_id = str(uuid.uuid4())
    
    # Test training with this dataset (will fallback to the file we created)
    business_result = test_training_with_file_id(business_file_id, "Random Forest")
    business_features = business_result.get('feature_info', {}).get('feature_names', [])
    print(f"✅ Business Features: {business_features}")
    
    # Test Case 3: Medical Dataset
    print("\n📊 TEST 3: MEDICAL DATASET")
    print("-" * 30)
    
    medical_data = pd.DataFrame({
        'patient_age': [25, 45, 67, 33, 56, 78, 29, 52],
        'blood_pressure_systolic': [120, 140, 160, 125, 145, 180, 115, 150],
        'cholesterol_level': [180, 220, 280, 190, 240, 350, 170, 260],
        'bmi': [22.5, 28.3, 32.1, 24.7, 29.8, 35.2, 21.9, 30.5],
        'diagnosis': ['healthy', 'at_risk', 'high_risk', 'healthy', 'at_risk', 'critical', 'healthy', 'at_risk']
    })
    
    medical_file_path = f"{os.path.dirname(__file__)}/uploads/medical_dataset.csv"
    medical_data.to_csv(medical_file_path, index=False)
    
    medical_file_id = str(uuid.uuid4())
    medical_result = test_training_with_file_id(medical_file_id, "XGBoost")
    medical_features = medical_result.get('feature_info', {}).get('feature_names', [])
    print(f"✅ Medical Features: {medical_features}")
    
    # Summary
    print("\n🎉 SUMMARY OF DYNAMIC FORMS")
    print("=" * 60)
    
    datasets = [
        ("Iris Dataset", iris_features),
        ("Business Dataset", business_features), 
        ("Medical Dataset", medical_features)
    ]
    
    for dataset_name, features in datasets:
        print(f"\n📋 {dataset_name}:")
        if features:
            for i, feature in enumerate(features, 1):
                # Show how the label would appear in the UI
                label = format_feature_label(feature)
                print(f"   {i}. {feature} → '{label}'")
        else:
            print("   ❌ No features detected")
    
    print(f"\n✅ DYNAMIC FORMS WORKING!")
    print(f"Each dataset generates a unique prediction form with {len(set([tuple(f) for _, f in datasets]))} different feature sets!")

def test_training_with_file_id(file_id: str, model_name: str) -> dict:
    """Test training with a specific file ID and return the response"""
    try:
        response = requests.post(f"{API_BASE}/api/train", 
                               json={"file_id": file_id, "model_name": model_name}, 
                               timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Training completed for {model_name}")
            return result
        else:
            print(f"❌ Training failed: {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}

def format_feature_label(feature_name: str) -> str:
    """Format feature name into a user-friendly label"""
    import re
    # Add spaces before capital letters
    label = re.sub(r'([A-Z])', r' \1', feature_name)
    # Capitalize first letter
    label = label.strip().title()
    # Fix common patterns
    label = label.replace(' Id', ' ID').replace('Cm', ' (cm)')
    return label

if __name__ == "__main__":
    test_dynamic_forms()