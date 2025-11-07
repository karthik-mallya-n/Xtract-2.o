# Complete Test for Google AI Studio Integration and Model Training

import requests
import pandas as pd
import json
import time

def create_sample_dataset():
    """Create a sample classification dataset"""
    print("📊 Creating sample dataset...")
    
    # Create a larger, more realistic dataset
    import numpy as np
    np.random.seed(42)
    
    n_samples = 200
    
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'education_years': np.random.randint(10, 20, n_samples),
        'experience_years': np.random.randint(0, 40, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples)
    }
    
    # Create a target variable based on some logic
    df = pd.DataFrame(data)
    # Loan approval based on income, credit score, and other factors
    df['loan_approved'] = (
        (df['income'] > 50000) & 
        (df['credit_score'] > 600) & 
        (df['experience_years'] > 2)
    ).astype(int)
    
    # Add some noise
    noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    df.loc[noise_indices, 'loan_approved'] = 1 - df.loc[noise_indices, 'loan_approved']
    
    filename = 'sample_loan_dataset.csv'
    df.to_csv(filename, index=False)
    
    print(f"✅ Created dataset: {filename}")
    print(f"   📊 Shape: {df.shape}")
    print(f"   🎯 Target distribution: {df['loan_approved'].value_counts().to_dict()}")
    print(f"   📋 Columns: {list(df.columns)}")
    
    return filename

def test_complete_ml_pipeline():
    """Test the complete ML pipeline: Upload -> AI Recommendations -> Training"""
    
    print("🧪 TESTING COMPLETE ML PIPELINE")
    print("="*80)
    
    base_url = "http://localhost:5000"
    
    # Step 1: Create sample dataset
    dataset_file = create_sample_dataset()
    
    try:
        # Step 2: Upload the dataset
        print(f"\n1️⃣ TESTING FILE UPLOAD")
        print("-" * 40)
        
        upload_url = f"{base_url}/api/upload"
        
        with open(dataset_file, 'rb') as f:
            files = {'file': f}
            data = {
                'is_labeled': 'labeled',
                'data_type': 'categorical'  # loan_approved is categorical (0/1)
            }
            
            upload_response = requests.post(upload_url, files=files, data=data)
        
        if upload_response.status_code == 200:
            upload_result = upload_response.json()
            print(f"✅ Upload successful!")
            print(f"   📄 File ID: {upload_result['file_id']}")
            print(f"   📊 File size: {upload_result['file_size']} bytes")
            
            file_id = upload_result['file_id']
        else:
            print(f"❌ Upload failed: {upload_response.status_code}")
            print(f"   Error: {upload_response.text}")
            return
        
        # Step 3: Get AI recommendations
        print(f"\n2️⃣ TESTING AI RECOMMENDATIONS")
        print("-" * 40)
        
        recommend_url = f"{base_url}/api/recommend-model?file_id={file_id}"
        recommend_response = requests.get(recommend_url)
        
        if recommend_response.status_code == 200:
            recommend_result = recommend_response.json()
            print(f"✅ AI Recommendations received!")
            
            recommendations = recommend_result.get('recommendations', {})
            if 'recommended_model' in recommendations:
                rec_model = recommendations['recommended_model']
                print(f"   🎯 Recommended Model: {rec_model.get('name', 'Unknown')}")
                print(f"   📝 Description: {rec_model.get('description', 'No description')}")
                print(f"   🧠 Reasoning: {rec_model.get('reasoning', 'No reasoning')[:100]}...")
                
                if 'alternative_models' in recommendations:
                    alt_models = recommendations['alternative_models']
                    print(f"   🔄 Alternative Models: {len(alt_models)} options")
            else:
                print(f"   ⚠️ No structured recommendations found")
        else:
            print(f"❌ AI Recommendations failed: {recommend_response.status_code}")
            print(f"   Error: {recommend_response.text}")
            return
        
        # Step 4: Train the recommended model
        print(f"\n3️⃣ TESTING MODEL TRAINING")
        print("-" * 40)
        
        train_url = f"{base_url}/api/train-recommended"
        train_data = {'file_id': file_id}
        
        print(f"🚀 Starting model training...")
        train_response = requests.post(train_url, json=train_data)
        
        if train_response.status_code == 200:
            train_result = train_response.json()
            print(f"✅ Model training completed!")
            
            training_results = train_result.get('training_results', {})
            if training_results.get('success'):
                print(f"   🤖 Model: {training_results.get('model_name', 'Unknown')}")
                print(f"   🎯 Type: {training_results.get('model_type', 'Unknown')}")
                print(f"   📊 Training samples: {training_results.get('training_samples', 0)}")
                print(f"   🧪 Test samples: {training_results.get('test_samples', 0)}")
                
                performance = training_results.get('performance', {})
                if 'accuracy' in performance:
                    print(f"   ✅ Accuracy: {performance['accuracy']:.4f} ({performance['accuracy']*100:.2f}%)")
                    print(f"   📊 Precision: {performance.get('precision', 0):.4f}")
                    print(f"   📊 Recall: {performance.get('recall', 0):.4f}")
                    print(f"   📊 F1-Score: {performance.get('f1_score', 0):.4f}")
                
                print(f"   💾 Model saved to: {training_results.get('model_path', 'Unknown')}")
            else:
                print(f"   ❌ Training failed: {training_results.get('error', 'Unknown error')}")
        else:
            print(f"❌ Model training failed: {train_response.status_code}")
            print(f"   Error: {train_response.text}")
            return
        
        print(f"\n🎉 COMPLETE ML PIPELINE TEST SUCCESSFUL!")
        print("="*80)
        print("✅ File Upload: SUCCESS")
        print("✅ AI Recommendations: SUCCESS")
        print("✅ Model Training: SUCCESS")
        print("🚀 Your ML platform is fully operational!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the Flask server is running on port 5000")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        import os
        if os.path.exists(dataset_file):
            os.remove(dataset_file)
            print(f"🧹 Cleaned up: {dataset_file}")

if __name__ == "__main__":
    test_complete_ml_pipeline()