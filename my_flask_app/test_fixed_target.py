#!/usr/bin/env python3
"""
Test with Manual Target Column
"""

import requests
import json

def test_with_correct_target():
    """Test training with manually specified target column"""
    
    print("🧪 TESTING WITH CORRECT TARGET COLUMN")
    print("=" * 50)
    
    # Use the Walmart dataset but train with Weekly_Sales as target
    file_id = "a0c0f494-6319-4a34-a708-f09796a60f92"
    model_name = "Lasso Regression"
    
    print(f"📊 File ID: {file_id}")
    print(f"🤖 Model: {model_name}")
    
    # For this test, let's directly modify the target in the dataset
    # and create a training call that uses Weekly_Sales as target
    
    # First, let's check what the current CSV looks like
    import pandas as pd
    
    df = pd.read_csv(f"uploads/{file_id}.csv")
    print(f"\n📋 Dataset Info:")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    
    # Let's try a different approach - modify the training call to use Weekly_Sales as target
    # by temporarily reordering the columns so Weekly_Sales is last
    
    # Reorder columns to put Weekly_Sales at the end (so auto-detection picks it as target)
    columns_reordered = [col for col in df.columns if col != 'Weekly_Sales'] + ['Weekly_Sales']
    df_reordered = df[columns_reordered]
    
    # Save as a new file
    new_file_id = "walmart_fixed_target"
    new_file_path = f"uploads/{new_file_id}.csv"
    df_reordered.to_csv(new_file_path, index=False)
    
    print(f"✅ Created fixed dataset with Weekly_Sales as last column")
    print(f"New columns order: {list(df_reordered.columns)}")
    
    # Now test training with the reordered dataset
    payload = {
        "file_id": new_file_id,
        "model_name": model_name
    }
    
    try:
        response = requests.post("http://localhost:5000/api/train", json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ SUCCESS!")
            
            feature_info = result.get('feature_info', {})
            if feature_info:
                print(f"\n🎉 CORRECTED FEATURE INFO:")
                features = feature_info.get('feature_names', [])
                target = feature_info.get('target_column', 'Unknown')
                
                print(f"📊 Features: {features}")
                print(f"🎯 Target: {target}")
                
                if target == 'Weekly_Sales' and 'Weekly_Sales' not in features:
                    print(f"✅ PERFECT! Target is Weekly_Sales and not in features")
                elif target == 'Weekly_Sales' and 'Weekly_Sales' in features:
                    print(f"❌ ERROR: Target Weekly_Sales still in features")
                else:
                    print(f"⚠️ Target is {target}, expected Weekly_Sales")
                    
                print(f"\n📋 Features for prediction form:")
                for i, feature in enumerate(features, 1):
                    print(f"   {i}. {feature}")
                    
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_with_correct_target()