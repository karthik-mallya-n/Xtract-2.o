"""
Core Machine Learning Module
Contains the main ML logic for model recommendations and data processing.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import joblib

# Import the advanced model trainer
from advanced_model_trainer import AdvancedModelTrainer

# Load environment variables
load_dotenv()

class MLCore:
    """Core ML functionality for the platform"""
    
    def __init__(self):
        # Google AI Studio configuration
        self.google_api_key = os.getenv('GOOGLE_AI_API_KEY')
        if not self.google_api_key:
            raise ValueError("GOOGLE_AI_API_KEY not found in environment variables")
        
        # Configure Google AI Studio
        genai.configure(api_key=self.google_api_key)
        # Use the best available stable model for Google AI Studio
        self.genai_model = genai.GenerativeModel('models/gemini-2.5-flash')
        
        # Initialize the advanced model trainer
        self.advanced_trainer = AdvancedModelTrainer(base_models_dir="models")
        
        print(f"✅ MLCore initialized with Google AI Studio (Gemini 2.5 Flash)")
        print(f"🔑 API Key: {self.google_api_key[:10]}...{self.google_api_key[-4:]}")
        print(f"🤖 Advanced Model Trainer initialized")
    
    def analyze_dataset(self, file_path: str, sample_size: int = 20) -> Dict[str, Any]:
        """
        Analyze the uploaded dataset and extract key characteristics
        
        Args:
            file_path (str): Path to the uploaded dataset
            sample_size (int): Number of rows to sample for analysis
            
        Returns:
            Dict containing dataset analysis results
        """
        try:
            # Determine file type and load dataset
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            # Basic dataset information
            analysis = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'sample_data': df.head(sample_size).to_dict('records'),
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'first_20_rows_csv': df.head(20).to_csv(index=False),  # Add this missing field
            }
            
            # Statistical summary for numeric columns
            if analysis['numeric_columns']:
                analysis['numeric_summary'] = df[analysis['numeric_columns']].describe().to_dict()
            
            # Unique value counts for categorical columns
            categorical_info = {}
            for col in analysis['categorical_columns']:
                unique_count = df[col].nunique()
                categorical_info[col] = {
                    'unique_values': unique_count,
                    'sample_values': df[col].value_counts().head(5).to_dict()
                }
            analysis['categorical_summary'] = categorical_info
            
            return analysis
            
        except Exception as e:
            raise Exception(f"Error analyzing dataset: {str(e)}")
    
    def create_llm_prompt(self, user_answers: Dict[str, Any], dataset_analysis: Dict[str, Any]) -> str:
        """
        Create a detailed prompt for the LLM to recommend ML models
        
        Args:
            user_answers (dict): User's responses about their data
            dataset_analysis (dict): Analysis results from the dataset
            
        Returns:
            str: Formatted prompt for the LLM
        """
        
        # Extract key information
        is_labeled = user_answers.get('is_labeled', 'unknown')
        data_type = user_answers.get('data_type', 'unknown')
        
        # Build dataset summary
        dataset_summary = f"""
Dataset Overview:
- Total rows: {dataset_analysis['total_rows']}
- Total columns: {dataset_analysis['total_columns']}
- Numeric columns: {len(dataset_analysis['numeric_columns'])}
- Categorical columns: {len(dataset_analysis['categorical_columns'])}
- Missing values: {sum(dataset_analysis['missing_values'].values())} total

Column Information:
{json.dumps(dataset_analysis['data_types'], indent=2)}

Sample Data (first few rows):
{json.dumps(dataset_analysis['sample_data'][:5], indent=2)}
"""

        prompt = f"""
You are an expert machine learning consultant. Based on the user's data characteristics and dataset analysis, recommend the most suitable machine learning models.

User Input:
- Data is labeled: {is_labeled}
- Data type: {data_type}

{dataset_summary}

Please analyze this data and provide model recommendations in the following JSON format:

{{
    "recommended_model": {{
        "name": "Model Name",
        "description": "Detailed explanation of why this model is recommended",
        "expected_accuracy": 85,
        "reasoning": "Specific reasons why this model fits the data"
    }},
    "alternative_models": [
        {{
            "name": "Alternative Model 1",
            "description": "Brief description",
            "expected_accuracy": 80,
            "pros": ["Advantage 1", "Advantage 2"],
            "cons": ["Limitation 1", "Limitation 2"]
        }},
        {{
            "name": "Alternative Model 2", 
            "description": "Brief description",
            "expected_accuracy": 78,
            "pros": ["Advantage 1", "Advantage 2"],
            "cons": ["Limitation 1", "Limitation 2"]
        }}
    ],
    "data_preprocessing_suggestions": [
        "Suggestion 1",
        "Suggestion 2"
    ],
    "potential_challenges": [
        "Challenge 1",
        "Challenge 2"
    ]
}}

Consider the following factors in your recommendation:
1. Whether the data is labeled (supervised vs unsupervised learning)
2. The data type (continuous, categorical, mixed)
3. Dataset size and complexity
4. Potential data quality issues
5. Interpretability requirements
6. Performance expectations

Provide practical, actionable recommendations based on the actual data characteristics.
"""
        
        return prompt.strip()
    
    def make_llm_request(self, user_answers: Dict[str, Any], dataset_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make API request to Google AI Studio (Gemini) for model recommendations
        
        Args:
            user_answers (dict): User's questionnaire responses
            dataset_analysis (dict): Dataset analysis results
            
        Returns:
            dict: LLM response with model recommendations
        """
        try:
            print(f"\n🤖 MAKING REQUEST TO GOOGLE AI STUDIO")
            print("="*80)
            
            # Log the data being sent
            print(f"📤 DATA BEING SENT TO GOOGLE AI STUDIO:")
            print(f"   👤 User Answers: {user_answers}")
            print(f"   📊 Dataset Info: {dataset_analysis['total_rows']} rows, {dataset_analysis['total_columns']} columns")
            print(f"   🔢 Numeric columns: {dataset_analysis['numeric_columns']}")
            print(f"   📝 Categorical columns: {dataset_analysis['categorical_columns']}")
            print(f"   📋 First 20 rows length: {len(dataset_analysis['first_20_rows_csv'])} characters")
            
            # Create persona-based prompt
            target_type = "categorical" if user_answers.get('data_type') == 'categorical' else "continuous"
            is_labeled = user_answers.get('is_labeled', 'labeled')
            
            # Build comprehensive prompt with all models for detected scenario
            prompt_template = """You are an expert Machine Learning Engineer with 15+ years of experience in model selection and predictive analytics.

Your task is to analyze this dataset and provide comprehensive model recommendations based on the four fundamental scenarios in machine learning.

🎯 **Scenario 1: Labeled + Continuous (Regression)**
Task: Predict a continuous numerical value
ALL Models: Linear Regression, Lasso Regression, Ridge Regression, ElasticNet, Support Vector Regression (SVR), K-Nearest Neighbors (KNN) Regressor, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, XGBoost Regressor, LightGBM Regressor, CatBoost Regressor, Neural Networks (MLP Regressor)

🏷️ **Scenario 2: Labeled + Categorical (Classification)**  
Task: Predict a discrete class label
ALL Models: Logistic Regression, Support Vector Machines (SVM), K-Nearest Neighbors (KNN) Classifier, Naive Bayes, Decision Tree Classifier, Random Forest Classifier, Gradient Boosting Classifier, XGBoost Classifier, LightGBM Classifier, CatBoost Classifier, Neural Networks (MLP Classifier)

🧩 **Scenario 3: Unlabeled + Continuous (Clustering/Dimensionality Reduction)**
Task: Find hidden groups or simplify data
ALL Models: K-Means Clustering, DBSCAN, Hierarchical Clustering, Gaussian Mixture Model (GMM), Principal Component Analysis (PCA), t-SNE, UMAP, Isolation Forest, One-Class SVM

🔗 **Scenario 4: Unlabeled + Categorical (Clustering/Association Rules)**
Task: Group similar items or find association rules
ALL Models: K-Modes Clustering, Hierarchical Clustering (Hamming distance), Apriori Algorithm, FP-Growth Algorithm, Eclat Algorithm, Multiple Correspondence Analysis (MCA)

Dataset Information:
- Rows: {total_rows}
- Columns: {total_columns}  
- Target Variable Type: {target_type}
- Data Labeling: {is_labeled}
- Numeric Columns: {numeric_columns}
- Categorical Columns: {categorical_columns}

First 20 rows of actual data:
{first_20_rows_csv}

CRITICAL INSTRUCTIONS:
1. **Semantic Analysis**: Analyze column names and values to understand the dataset's domain
2. **Scenario Detection**: Determine which of the 4 scenarios applies to this dataset
3. **INCLUDE ALL MODELS**: From the detected scenario, include ALL models listed above (not just top 3)
4. **Rank by Accuracy**: Rank ALL models in descending order of expected accuracy for this specific dataset

You MUST include ALL models from the detected scenario's model list. Do not limit to just 3-5 models.

Respond in JSON format with ALL models:

```json
{{
  "scenario_detected": {{
    "type": "Labeled + Continuous | Labeled + Categorical | Unlabeled + Continuous | Unlabeled + Categorical",
    "task": "Regression | Classification | Clustering | Association Rules",
    "reasoning": "Why this scenario was selected based on data analysis"
  }},
  "semantic_analysis": {{
    "domain": "Identified domain (medical, financial, etc.)",
    "key_insights": "Important observations from column names and data patterns"
  }},
  "recommended_models": [
    {{
      "rank": 1,
      "name": "Model Name",
      "expected_accuracy": "95-98%",
      "reasoning": "Why this model is ranked #1 for this dataset",
      "advantages": "Key strengths for this scenario"
    }},
    {{
      "rank": 2,
      "name": "Model Name", 
      "expected_accuracy": "90-95%",
      "reasoning": "Why this model is ranked #2",
      "advantages": "Key strengths"
    }}
    // CONTINUE FOR ALL MODELS IN THE DETECTED SCENARIO - DO NOT STOP AT 3
  ],
  "primary_recommendation": {{
    "model": "Top ranked model name",
    "confidence": "High/Medium/Low",
    "rationale": "Final recommendation reasoning"
  }}
}}
```

REMEMBER: Include ALL models from the detected scenario, not just the top few. The user specifically wants to see the complete list ranked by accuracy."""

            # Format the prompt with actual data
            prompt = prompt_template.format(
                total_rows=dataset_analysis['total_rows'],
                total_columns=dataset_analysis['total_columns'],
                target_type=target_type,
                is_labeled=is_labeled,
                numeric_columns=dataset_analysis['numeric_columns'],
                categorical_columns=dataset_analysis['categorical_columns'],
                first_20_rows_csv=dataset_analysis['first_20_rows_csv']
            )
            
            print(f"📤 SENDING REQUEST TO GEMINI 2.5 FLASH")
            print(f"📋 Prompt length: {len(prompt)} characters")
            print(f"🔗 Model: Gemini 2.5 Flash")
            
            # Make the request to Gemini
            response = self.genai_model.generate_content(prompt)
            raw_response = response.text
            
            print(f"\n📥 RECEIVED RESPONSE FROM GOOGLE AI STUDIO:")
            print(f"   ✅ Response received successfully")
            print(f"   📏 Response length: {len(raw_response)} characters")
            
            # Enhanced logging for terminal
            print("\n" + "="*80)
            print("🤖 GEMINI MODEL RECOMMENDATION RESPONSE")
            print("="*80)
            print(f"Target Variable Type: {target_type.title()}")
            print(f"Dataset Columns: {dataset_analysis['total_columns']} columns")
            print(f"Dataset Rows: {dataset_analysis['total_rows']} rows")
            
            print(f"\n📋 RAW GEMINI RESPONSE:")
            print("-" * 50)
            print(raw_response)
            print("-" * 50)

            # Try to parse JSON response
            try:
                # Clean the response (remove markdown code blocks if present)
                cleaned_response = raw_response
                if "```json" in cleaned_response:
                    cleaned_response = cleaned_response.split("```json")[1]
                    if "```" in cleaned_response:
                        cleaned_response = cleaned_response.split("```")[0]
                elif "```" in cleaned_response:
                    # Handle cases with just ```
                    parts = cleaned_response.split("```")
                    if len(parts) >= 3:
                        cleaned_response = parts[1]
                
                parsed_recommendations = json.loads(cleaned_response.strip())
                
                print(f"\n✅ PARSED RECOMMENDATIONS:")
                print("-" * 50)
                
                # Display scenario detection
                if 'scenario_detected' in parsed_recommendations:
                    scenario = parsed_recommendations['scenario_detected']
                    print(f"🎯 SCENARIO: {scenario.get('type', 'Unknown')}")
                    print(f"📝 Task: {scenario.get('task', 'Unknown')}")
                    print(f"🧠 Reasoning: {scenario.get('reasoning', 'No reasoning')}")
                
                # Display semantic analysis
                if 'semantic_analysis' in parsed_recommendations:
                    semantic = parsed_recommendations['semantic_analysis']
                    print(f"\n🔍 SEMANTIC ANALYSIS:")
                    print(f"🏢 Domain: {semantic.get('domain', 'Unknown')}")
                    print(f"💡 Key Insights: {semantic.get('key_insights', 'No insights')}")
                
                # Display ranked models
                if 'recommended_models' in parsed_recommendations:
                    models = parsed_recommendations['recommended_models']
                    print(f"\n� RANKED MODELS (by expected accuracy):")
                    for model in models:
                        rank = model.get('rank', 'Unknown')
                        name = model.get('name', 'Unknown')
                        accuracy = model.get('expected_accuracy', 'Unknown')
                        print(f"  #{rank}. {name} - {accuracy}")
                        print(f"      💫 Reasoning: {model.get('reasoning', 'No reasoning')}")
                        print(f"      ✅ Advantages: {model.get('advantages', 'No advantages')}")
                        print()
                
                # Display primary recommendation
                if 'primary_recommendation' in parsed_recommendations:
                    primary = parsed_recommendations['primary_recommendation']
                    print(f"🏆 PRIMARY RECOMMENDATION: {primary.get('model', 'Unknown')}")
                    print(f"🎯 Confidence: {primary.get('confidence', 'Unknown')}")
                    print(f"📋 Rationale: {primary.get('rationale', 'No rationale')}")
                
                print("="*80)
                
                return {
                    'success': True,
                    'recommendations': parsed_recommendations,
                    'raw_response': raw_response
                }
                
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parsing failed: {str(e)}")
                print(f"📄 Returning raw response")
                return {
                    'success': True,
                    'recommendations': {},
                    'raw_response': raw_response,
                    'error': f'Failed to parse JSON: {str(e)}'
                }
            
        except Exception as e:
            print(f"❌ Error making request to Google AI Studio: {str(e)}")
            return {
                'success': False,
                'error': f'API request failed: {str(e)}',
                'recommendations': {},
                'raw_response': ''
            }
    
    def train_recommended_model(self, file_path: str, recommendations: Dict[str, Any], user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the recommended model based on AI recommendations
        
        Args:
            file_path (str): Path to the dataset
            recommendations (dict): AI model recommendations
            user_data (dict): User-provided data about the problem
            
        Returns:
            dict: Training results with model performance
        """
        try:
            print(f"\n🚀 STARTING MODEL TRAINING")
            print("="*80)
            
            # Load the dataset
            df = pd.read_csv(file_path)
            print(f"📊 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Get the recommended model
            recommended_model_info = recommendations.get('recommended_model', {})
            model_name = recommended_model_info.get('name', 'Random Forest Classifier')
            
            print(f"🎯 Training Model: {model_name}")
            print(f"📝 Reasoning: {recommended_model_info.get('reasoning', 'No reasoning provided')}")
            
            # Determine target variable (assume last column is target)
            target_column = df.columns[-1]
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            print(f"🎯 Target Variable: {target_column}")
            print(f"📊 Features: {list(X.columns)}")
            
            # Handle categorical variables (simple label encoding for now)
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = pd.Categorical(X[col]).codes
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"📊 Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
            
            # Determine problem type
            is_classification = user_data.get('data_type') == 'categorical' or len(y.unique()) < 20
            
            # Get the appropriate model
            model = self._get_model_instance(model_name, is_classification)
            
            print(f"🤖 Model Type: {'Classification' if is_classification else 'Regression'}")
            print(f"🔧 Model: {type(model).__name__}")
            
            # Train the model
            print(f"⏳ Training model...")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate performance metrics
            if is_classification:
                accuracy = accuracy_score(y_test, y_pred)
                print(f"✅ Training Complete!")
                print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                
                # Classification report
                report = classification_report(y_test, y_pred, output_dict=True)
                
                performance = {
                    'accuracy': accuracy,
                    'precision': report['macro avg']['precision'],
                    'recall': report['macro avg']['recall'],
                    'f1_score': report['macro avg']['f1-score'],
                    'classification_report': report
                }
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                print(f"✅ Training Complete!")
                print(f"📊 RMSE: {rmse:.4f}")
                print(f"📊 MSE: {mse:.4f}")
                
                performance = {
                    'rmse': rmse,
                    'mse': mse,
                    'r2_score': model.score(X_test, y_test)
                }
            
            # Save the model
            model_filename = f"trained_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            model_path = os.path.join('models', model_filename)
            os.makedirs('models', exist_ok=True)
            joblib.dump(model, model_path)
            
            print(f"💾 Model saved to: {model_path}")
            print("="*80)
            
            return {
                'success': True,
                'model_name': model_name,
                'model_type': 'classification' if is_classification else 'regression',
                'performance': performance,
                'model_path': model_path,
                'feature_names': list(X.columns),
                'target_column': target_column,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            print(f"❌ Error training model: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_model_instance(self, model_name: str, is_classification: bool):
        """
        Get an instance of the specified model
        
        Args:
            model_name (str): Name of the model
            is_classification (bool): Whether this is a classification problem
            
        Returns:
            sklearn model instance
        """
        model_name_lower = model_name.lower()
        
        if is_classification:
            if 'random forest' in model_name_lower:
                return RandomForestClassifier(n_estimators=100, random_state=42)
            elif 'logistic regression' in model_name_lower:
                return LogisticRegression(random_state=42, max_iter=1000)
            elif 'svm' in model_name_lower or 'support vector' in model_name_lower:
                return SVC(random_state=42)
            else:
                # Default to Random Forest for classification
                return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            if 'random forest' in model_name_lower:
                return RandomForestRegressor(n_estimators=100, random_state=42)
            elif 'linear regression' in model_name_lower:
                return LinearRegression()
            elif 'svm' in model_name_lower or 'support vector' in model_name_lower:
                return SVR()
            else:
                # Default to Random Forest for regression
                return RandomForestRegressor(n_estimators=100, random_state=42)

    def get_model_class(self, model_name: str):
        """
        Get the scikit-learn model class based on model name
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Scikit-learn model class
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.svm import SVC, SVR
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        
        # Map model names to classes (you can extend this)
        model_mapping = {
            'random_forest': RandomForestClassifier,
            'random_forest_classifier': RandomForestClassifier,
            'random_forest_regressor': RandomForestRegressor,
            'svm': SVC,
            'support_vector_machine': SVC,
            'svc': SVC,
            'svr': SVR,
            'logistic_regression': LogisticRegression,
            'linear_regression': LinearRegression,
            'neural_network': MLPClassifier,
            'mlp_classifier': MLPClassifier,
            'mlp_regressor': MLPRegressor,
            'naive_bayes': GaussianNB,
            'decision_tree': DecisionTreeClassifier,
            'decision_tree_classifier': DecisionTreeClassifier,
            'decision_tree_regressor': DecisionTreeRegressor,
        }
        
        # Normalize model name
        normalized_name = model_name.lower().replace(' ', '_').replace('-', '_')
        
        return model_mapping.get(normalized_name, RandomForestClassifier)
    
    def prepare_data_for_training(self, df: pd.DataFrame, target_column: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare dataset for model training
        
        Args:
            df (DataFrame): Input dataset
            target_column (str): Name of target column
            
        Returns:
            Tuple of (X, y) arrays
        """
        try:
            # If no target column specified, try to infer it
            if target_column is None:
                # Assume last column is target, or look for common target names
                possible_targets = ['target', 'label', 'class', 'y', 'output']
                for col in possible_targets:
                    if col in df.columns:
                        target_column = col
                        break
                else:
                    # Use last column as target
                    target_column = df.columns[-1]
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Handle categorical variables
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            
            # Encode categorical features
            label_encoders = {}
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
            
            # Encode target if categorical
            target_encoder = None
            if y.dtype == 'object':
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y.astype(str))
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            return X_scaled, y, {
                'feature_names': X.columns.tolist(),
                'target_name': target_column,
                'label_encoders': label_encoders,
                'target_encoder': target_encoder,
                'scaler': scaler
            }
            
        except Exception as e:
            raise Exception(f"Error preparing data for training: {str(e)}")
    
    def train_advanced_model(self, model_name: str, file_path: str, target_column: str) -> Dict[str, Any]:
        """
        Train a model using the advanced trainer with 90%+ accuracy optimization
        
        Args:
            model_name (str): Name of the model to train
            file_path (str): Path to the dataset file
            target_column (str): Name of the target column
            
        Returns:
            dict: Training results with performance metrics
        """
        try:
            print(f"\n🚀 STARTING ADVANCED MODEL TRAINING")
            print("="*80)
            print(f"🤖 Model: {model_name}")
            print(f"📄 Dataset: {file_path}")
            print(f"🎯 Target: {target_column}")
            
            # Use the advanced trainer
            result = self.advanced_trainer.train_model(
                model_name=model_name,
                file_path=file_path,
                target_column=target_column
            )
            
            if result['success']:
                print(f"\n🎉 ADVANCED TRAINING COMPLETED SUCCESSFULLY!")
                print(f"📁 Model folder: {result['model_folder']}")
                print(f"🎯 {result['score_name']}: {result['main_score']:.4f} ({result['main_score']*100:.2f}%)")
                
                if result['threshold_met']:
                    print(f"✅ SUCCESS: Achieved 90%+ performance!")
                else:
                    print(f"⚠️  Below 90% threshold but model trained successfully")
            else:
                print(f"❌ ADVANCED TRAINING FAILED: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in advanced model training: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name
            }
    
    def get_available_models(self, problem_type: str = None) -> List[str]:
        """
        Get list of available models for training
        
        Args:
            problem_type (str): 'classification' or 'regression' or None for all
            
        Returns:
            list: Available model names
        """
        if problem_type == 'classification':
            return list(self.advanced_trainer.classification_models.keys())
        elif problem_type == 'regression':
            return list(self.advanced_trainer.regression_models.keys())
        else:
            # Return all models
            all_models = list(self.advanced_trainer.classification_models.keys())
            all_models.extend(list(self.advanced_trainer.regression_models.keys()))
            return all_models
    
    def predict_with_model(self, model_folder: str, new_data: pd.DataFrame, timestamp: str = None) -> np.ndarray:
        """
        Make predictions using a trained model
        
        Args:
            model_folder (str): Path to the model folder
            new_data (pd.DataFrame): New data for prediction
            timestamp (str): Specific model timestamp (optional)
            
        Returns:
            np.ndarray: Predictions
        """
        return self.advanced_trainer.predict(model_folder, new_data, timestamp)

# Create a global instance
ml_core = MLCore()