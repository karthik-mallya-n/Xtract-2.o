#!/usr/bin/env python3
"""
Comprehensive Test Script for Column Filtering Workflow

This script tests the complete workflow:
1. Upload a CSV file
2. Select specific columns (not all)
3. Start training
4. Verify that only selected columns are used in training and results
"""

import requests
import json
import time
import os

# Configuration
BASE_URL = "http://127.0.0.1:5000/api"
TEST_FILE_PATH = r"e:\New Codes\MP 2.o\02\test_dataset.csv"

def test_column_filtering_workflow():
    """Test the complete column filtering workflow"""
    
    print("=" * 100)
    print("🚀 COMPREHENSIVE COLUMN FILTERING TEST")
    print("=" * 100)
    
    # Read available columns from the CSV file for reference
    import pandas as pd
    df = pd.read_csv(TEST_FILE_PATH)
    available_columns = list(df.columns)
    print(f"📊 Available columns in test file: {available_columns}")
    print(f"📊 Total available columns: {len(available_columns)}")
    
    # Step 1: Upload the test file with column selection
    print("\n1️⃣ UPLOADING TEST FILE WITH COLUMN SELECTION")
    print("-" * 50)
    
    if not os.path.exists(TEST_FILE_PATH):
        print(f"❌ Test file not found: {TEST_FILE_PATH}")
        return False
    
    # Step 2: Define column selection beforehand (including ID column to test filtering)
    selected_columns = ["Store", "Weekly_Sales", "Temperature", "Holiday_Flag"]  # Include Store to test ID filtering
    
    print(f"📋 Selected columns for upload: {selected_columns}")
    print(f"💡 Note: 'Store' is an ID column and should be automatically filtered out during training")
    
    with open(TEST_FILE_PATH, 'rb') as f:
        files = {'file': f}
        data = {
            'is_labeled': 'unlabeled',  # Required form field
            'selected_columns': json.dumps(selected_columns)  # Include column selection
        }
        response = requests.post(f"{BASE_URL}/upload", files=files, data=data)
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    upload_data = response.json()
    file_id = upload_data.get('file_id')
    
    print(f"✅ File uploaded successfully")
    print(f"📄 File ID: {file_id}")
    print(f"📋 Selected columns: {selected_columns}")
    print(f"📋 Selected count: {len(selected_columns)}")
    
    # Step 3: Set up training data (no need for separate update since we included in upload)
    print(f"\n2️⃣ PREPARING FOR TRAINING")
    print("-" * 50)
    print(f"📋 Using selected columns from upload: {selected_columns}")
    print(f"📋 Excluded columns: {[col for col in available_columns if col not in selected_columns]}")
    
    # Step 4: Start training with K-Means (unsupervised)
    print(f"\n3️⃣ STARTING TRAINING")
    print("-" * 50)
    
    training_data = {
        'file_id': file_id,
        'model_name': 'K-Means'
    }
    
    print(f"🚀 Training with: {training_data['model_name']}")
    print(f"📊 Expected to use only: {selected_columns}")
    
    training_response = requests.post(f"{BASE_URL}/train", 
                                     json=training_data,
                                     timeout=300)  # 5 minute timeout
    
    if training_response.status_code != 200:
        print(f"❌ Training failed: {training_response.status_code}")
        print(f"Response: {training_response.text}")
        return False
    
    training_result = training_response.json()
    
    print(f"✅ Training completed!")
    print(f"📊 Training success: {training_result.get('success')}")
    
    # Step 5: Analyze the results
    print(f"\n4️⃣ ANALYZING RESULTS")
    print("-" * 50)
    
    # Extract feature_info from the response
    feature_info = training_result.get('feature_info', {})
    
    print(f"📋 Feature info keys: {list(feature_info.keys())}")
    
    # Check feature names used in training
    feature_names_used = feature_info.get('feature_names', [])
    print(f"📊 Feature names used in training: {feature_names_used}")
    print(f"📊 Number of features used: {len(feature_names_used)}")
    
    # Check selected_columns in feature_info
    selected_columns_in_result = feature_info.get('selected_columns', [])
    print(f"📋 Selected columns in result: {selected_columns_in_result}")
    print(f"📋 Selected columns count in result: {len(selected_columns_in_result)}")
    
    # Step 6: Validate results
    print(f"\n5️⃣ VALIDATION")
    print("-" * 50)
    
    all_tests_passed = True
    
    # Test 1: Check if actual training features make sense (ID columns should be filtered)
    expected_features_after_filtering = [col for col in selected_columns if col != "Store"]  # Store should be filtered out
    if set(selected_columns_in_result) == set(expected_features_after_filtering):
        print(f"✅ TEST 1 PASSED: ID column filtering worked correctly")
        print(f"   Original selection: {sorted(selected_columns)}")
        print(f"   After ID filtering: {sorted(expected_features_after_filtering)}")
        print(f"   Result: {sorted(selected_columns_in_result)}")
    else:
        print(f"❌ TEST 1 FAILED: ID column filtering didn't work as expected")
        print(f"   Original selection: {sorted(selected_columns)}")
        print(f"   Expected after filtering: {sorted(expected_features_after_filtering)}")
        print(f"   Got: {sorted(selected_columns_in_result)}")
        all_tests_passed = False
    
    # Test 2: Check if training used the correct filtered features
    if set(feature_names_used) == set(expected_features_after_filtering):
        print(f"✅ TEST 2 PASSED: Training used correct filtered features")
        print(f"   Expected: {sorted(expected_features_after_filtering)}")
        print(f"   Used: {sorted(feature_names_used)}")
    else:
        print(f"❌ TEST 2 FAILED: Training didn't use expected features")
        print(f"   Expected: {sorted(expected_features_after_filtering)}")
        print(f"   Used: {sorted(feature_names_used)}")
        all_tests_passed = False
    
    # Test 3: Check if excluded columns are NOT in feature_names
    excluded_columns = [col for col in available_columns if col not in selected_columns]
    features_in_excluded = [col for col in feature_names_used if col in excluded_columns]
    if not features_in_excluded:
        print(f"✅ TEST 3 PASSED: No excluded columns in training features")
        print(f"   Excluded: {excluded_columns}")
        print(f"   Training features: {feature_names_used}")
    else:
        print(f"❌ TEST 3 FAILED: Excluded columns found in training features: {features_in_excluded}")
        all_tests_passed = False
    
    # Test 4: Check training result structure
    if training_result.get('success') and 'result' in training_result:
        print(f"✅ TEST 4 PASSED: Training result structure is correct")
    else:
        print(f"❌ TEST 4 FAILED: Training result structure is incorrect")
        all_tests_passed = False
    
    # Test 5: Check that ID column (Store) was correctly filtered out
    if "Store" in selected_columns and "Store" not in feature_names_used:
        print(f"✅ TEST 5 PASSED: ID column 'Store' was correctly filtered out")
    elif "Store" not in selected_columns:
        print(f"✅ TEST 5 PASSED: No ID column in selection")
    else:
        print(f"❌ TEST 5 FAILED: ID column 'Store' should have been filtered out")
        all_tests_passed = False
    
    # Final Summary
    print(f"\n{'=' * 100}")
    if all_tests_passed:
        print(f"🎉 ALL TESTS PASSED! Column filtering is working correctly!")
        print(f"✅ Original selection: {selected_columns}")
        print(f"✅ After ID filtering: {selected_columns_in_result}")
        print(f"✅ Features used in training: {feature_names_used}")
        print(f"✅ ID column 'Store' correctly filtered out")
        print(f"✅ Only business-relevant features used for training")
    else:
        print(f"❌ SOME TESTS FAILED! Column filtering needs attention!")
        print(f"⚠️ Original columns: {available_columns}")
        print(f"⚠️ Selected columns: {selected_columns}")
        print(f"⚠️ Features used: {feature_names_used}")
        print(f"⚠️ Selected columns in result: {selected_columns_in_result}")
    print(f"{'=' * 100}")
    
    return all_tests_passed

if __name__ == "__main__":
    try:
        success = test_column_filtering_workflow()
        if success:
            print(f"\n🎯 WORKFLOW TEST COMPLETED SUCCESSFULLY!")
        else:
            print(f"\n🚨 WORKFLOW TEST FAILED!")
    except Exception as e:
        print(f"\n💥 TEST SCRIPT ERROR: {str(e)}")
        import traceback
        traceback.print_exc()