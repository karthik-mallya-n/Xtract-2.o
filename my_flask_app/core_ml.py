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
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, mean_absolute_error, r2_score
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
        
        # Generation configuration for complete responses
        self.generation_config = {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 8192,  # Maximum to prevent truncation
            "response_mime_type": "text/plain",
        }
        
        # Use the best available stable model for Google AI Studio
        self.genai_model = genai.GenerativeModel(
            model_name='models/gemini-2.5-flash',
            generation_config=self.generation_config
        )
        
        # Initialize the advanced model trainer
        self.advanced_trainer = AdvancedModelTrainer(base_models_dir="models")
        
        print(f"Γ£à MLCore initialized with Google AI Studio (Gemini 2.5 Flash)")
        print(f"≡ƒöæ API Key: {self.google_api_key[:10]}...{self.google_api_key[-4:]}")
        print(f"≡ƒñû Advanced Model Trainer initialized")
    
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
            print(f"\n≡ƒñû MAKING REQUEST TO GOOGLE AI STUDIO")
            print("="*80)
            
            # Log the data being sent
            print(f"≡ƒôñ DATA BEING SENT TO GOOGLE AI STUDIO:")
            print(f"   ≡ƒæñ User Answers: {user_answers}")
            print(f"   ≡ƒôè Dataset Info: {dataset_analysis['total_rows']} rows, {dataset_analysis['total_columns']} columns")
            print(f"   ≡ƒöó Numeric columns: {dataset_analysis['numeric_columns']}")
            print(f"   ≡ƒô¥ Categorical columns: {dataset_analysis['categorical_columns']}")
            print(f"   ≡ƒôï First 20 rows length: {len(dataset_analysis['first_20_rows_csv'])} characters")
            
            # Create persona-based prompt
            target_type = "categorical" if user_answers.get('data_type') == 'categorical' else "continuous"
            is_labeled = user_answers.get('is_labeled', 'labeled')
            
            # Build comprehensive prompt with all models for detected scenario
            prompt_template = """You are an expert Machine Learning Engineer with 15+ years of experience in model selection and predictive analytics.

Your task is to analyze this dataset and provide comprehensive model recommendations based on the four fundamental scenarios in machine learning.

≡ƒÄ» **Scenario 1: Labeled + Continuous (Regression)**
Task: Predict a continuous numerical value
ALL Models: Linear Regression, Lasso Regression, Ridge Regression, ElasticNet, Support Vector Regression (SVR), K-Nearest Neighbors (KNN) Regressor, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, XGBoost Regressor, LightGBM Regressor, CatBoost Regressor, Neural Networks (MLP Regressor)

≡ƒÅ╖∩╕Å **Scenario 2: Labeled + Categorical (Classification)**  
Task: Predict a discrete class label
ALL Models: Logistic Regression, Support Vector Machines (SVM), K-Nearest Neighbors (KNN) Classifier, Naive Bayes, Decision Tree Classifier, Random Forest Classifier, Gradient Boosting Classifier, XGBoost Classifier, LightGBM Classifier, CatBoost Classifier, Neural Networks (MLP Classifier)

≡ƒº⌐ **Scenario 3: Unlabeled + Continuous (Clustering/Dimensionality Reduction)**
Task: Find hidden groups or simplify data
ALL Models: K-Means Clustering, DBSCAN, Hierarchical Clustering, Gaussian Mixture Model (GMM), Principal Component Analysis (PCA), t-SNE, UMAP, Isolation Forest, One-Class SVM

≡ƒöù **Scenario 4: Unlabeled + Categorical (Clustering/Association Rules)**
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
            
            print(f"≡ƒôñ SENDING REQUEST TO GEMINI 2.5 FLASH")
            print(f"≡ƒôï Prompt length: {len(prompt)} characters")
            print(f"≡ƒöù Model: Gemini 2.5 Flash (Optimized for Complete Responses)")
            print(f"≡ƒôÅ Max Output Tokens: {self.generation_config['max_output_tokens']}")
            
            # Make the request to Gemini with full configuration
            response = self.genai_model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            raw_response = response.text
            
            print(f"\n≡ƒôÑ RECEIVED RESPONSE FROM GOOGLE AI STUDIO:")
            print(f"   Γ£à Response received successfully")
            print(f"   ≡ƒôÅ Response length: {len(raw_response)} characters")
            
            # Enhanced logging for terminal
            print("\n" + "="*80)
            print("≡ƒñû GEMINI MODEL RECOMMENDATION RESPONSE")
            print("="*80)
            print(f"Target Variable Type: {target_type.title()}")
            print(f"Dataset Columns: {dataset_analysis['total_columns']} columns")
            print(f"Dataset Rows: {dataset_analysis['total_rows']} rows")
            
            print(f"\n≡ƒôï RAW GEMINI RESPONSE:")
            print("-" * 50)
            print(raw_response)
            print("-" * 50)

            # Try to parse JSON response
            try:
                # Clean and parse the response
                cleaned_response = raw_response.strip()
                
                # Check if response seems incomplete (no closing brace)
                if not cleaned_response.endswith('}'):
                    print(f"ΓÜá∩╕Å Response appears incomplete - missing closing brace")
                    print(f"≡ƒôÅ Response ends with: '...{cleaned_response[-50:]}'")
                
                # Remove markdown code blocks if present
                if "```json" in cleaned_response:
                    if cleaned_response.count("```") >= 2:
                        # Extract content between ```json and ```
                        start_idx = cleaned_response.find("```json") + 7
                        end_idx = cleaned_response.find("```", start_idx)
                        if end_idx != -1:
                            cleaned_response = cleaned_response[start_idx:end_idx].strip()
                    else:
                        # Handle incomplete markdown - remove ```json from the start
                        cleaned_response = cleaned_response.replace("```json", "").strip()
                        # Remove any trailing ``` if present
                        if "```" in cleaned_response:
                            cleaned_response = cleaned_response.split("```")[0]
                elif "```" in cleaned_response:
                    # Handle cases with just ```
                    parts = cleaned_response.split("```")
                    if len(parts) >= 3:
                        cleaned_response = parts[1]
                
                # Try to fix incomplete JSON by adding missing parts
                if cleaned_response and not cleaned_response.endswith('}'):
                    print(f"≡ƒöº Attempting to fix incomplete JSON...")
                    
                    # Fix incomplete expected_accuracy field (most common truncation)
                    if '"expected_accuracy": "' in cleaned_response:
                        # Find the last occurrence
                        last_accuracy_idx = cleaned_response.rfind('"expected_accuracy": "')
                        if last_accuracy_idx != -1:
                            remaining_text = cleaned_response[last_accuracy_idx + 21:]  # After '"expected_accuracy": "'
                            # Check if the string is incomplete (no closing quote)
                            if '"' not in remaining_text or (remaining_text.find('"') == -1):
                                # Complete the accuracy field
                                if '%' not in remaining_text:
                                    cleaned_response += '%"'
                                else:
                                    cleaned_response += '"'
                                print(f"≡ƒöº Fixed incomplete accuracy field")
                    
                    # Fix incomplete reasoning field
                    if '"reasoning": "' in cleaned_response:
                        reasoning_count = cleaned_response.count('"reasoning": "')
                        quote_after_reasoning = cleaned_response.count('"reasoning": "') 
                        # Count how many reasoning fields are properly closed
                        closed_reasoning = cleaned_response.count('"reasoning":')
                        if reasoning_count > closed_reasoning:
                            cleaned_response += '"'
                            print(f"≡ƒöº Fixed incomplete reasoning field")
                    
                    # Remove incomplete trailing entry if it exists
                    lines = cleaned_response.split('\n')
                    if lines and not lines[-1].strip().endswith(('}', ']', '"')):
                        # Remove the incomplete last line
                        lines = lines[:-1]
                        cleaned_response = '\n'.join(lines)
                        print(f"≡ƒöº Removed incomplete final entry")
                    
                    # Ensure proper array closure
                    if '"recommended_models": [' in cleaned_response and not cleaned_response.count('[') == cleaned_response.count(']'):
                        if not cleaned_response.rstrip().endswith(']'):
                            # Close the array properly
                            if cleaned_response.rstrip().endswith(','):
                                cleaned_response = cleaned_response.rstrip()[:-1]  # Remove trailing comma
                            cleaned_response += '\n  ]'
                            print(f"≡ƒöº Fixed incomplete array")
                    
                    # Count opening vs closing braces to determine how many we need
                    open_braces = cleaned_response.count('{')
                    close_braces = cleaned_response.count('}')
                    missing_braces = open_braces - close_braces
                    
                    if missing_braces > 0:
                        # Add missing closing braces
                        cleaned_response += '}' * missing_braces
                        print(f"≡ƒöº Added {missing_braces} missing closing brace(s)")
                
                parsed_recommendations = json.loads(cleaned_response.strip())
                
                print(f"\nΓ£à PARSED RECOMMENDATIONS:")
                print("-" * 50)
                
                # Display scenario detection
                if 'scenario_detected' in parsed_recommendations:
                    scenario = parsed_recommendations['scenario_detected']
                    print(f"≡ƒÄ» SCENARIO: {scenario.get('type', 'Unknown')}")
                    print(f"≡ƒô¥ Task: {scenario.get('task', 'Unknown')}")
                    print(f"≡ƒºá Reasoning: {scenario.get('reasoning', 'No reasoning')}")
                
                # Display semantic analysis
                if 'semantic_analysis' in parsed_recommendations:
                    semantic = parsed_recommendations['semantic_analysis']
                    print(f"\n≡ƒöì SEMANTIC ANALYSIS:")
                    print(f"≡ƒÅó Domain: {semantic.get('domain', 'Unknown')}")
                    print(f"≡ƒÆí Key Insights: {semantic.get('key_insights', 'No insights')}")
                
                # Display ranked models
                if 'recommended_models' in parsed_recommendations:
                    models = parsed_recommendations['recommended_models']
                    print(f"\n∩┐╜ RANKED MODELS (by expected accuracy):")
                    for model in models:
                        rank = model.get('rank', 'Unknown')
                        name = model.get('name', 'Unknown')
                        accuracy = model.get('expected_accuracy', 'Unknown')
                        print(f"  #{rank}. {name} - {accuracy}")
                        print(f"      ≡ƒÆ½ Reasoning: {model.get('reasoning', 'No reasoning')}")
                        print(f"      Γ£à Advantages: {model.get('advantages', 'No advantages')}")
                        print()
                
                # Display primary recommendation
                if 'primary_recommendation' in parsed_recommendations:
                    primary = parsed_recommendations['primary_recommendation']
                    print(f"≡ƒÅå PRIMARY RECOMMENDATION: {primary.get('model', 'Unknown')}")
                    print(f"≡ƒÄ» Confidence: {primary.get('confidence', 'Unknown')}")
                    print(f"≡ƒôï Rationale: {primary.get('rationale', 'No rationale')}")
                
                print("="*80)
                
                return {
                    'success': True,
                    'recommendations': parsed_recommendations,
                    'raw_response': raw_response
                }
                
            except json.JSONDecodeError as e:
                print(f"ΓÜá∩╕Å JSON parsing failed: {str(e)}")
                print(f"≡ƒôä Returning raw response")
                return {
                    'success': True,
                    'recommendations': {},
                    'raw_response': raw_response,
                    'error': f'Failed to parse JSON: {str(e)}'
                }
            
        except Exception as e:
            print(f"Γ¥î Error making request to Google AI Studio: {str(e)}")
            return {
                'success': False,
                'error': f'API request failed: {str(e)}',
                'recommendations': {},
                'raw_response': ''
            }
    
    def train_recommended_model(self, file_path: str, recommendations: Dict[str, Any], user_data: Dict[str, Any], selected_columns: List[str] = None) -> Dict[str, Any]:
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
            print(f"\n≡ƒÜÇ STARTING MODEL TRAINING")
            print("="*80)
            
            # Load the dataset
            df = pd.read_csv(file_path)
            print(f"≡ƒôè Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Get the recommended model
            recommended_model_info = recommendations.get('recommended_model', {})
            model_name = recommended_model_info.get('name', 'Random Forest Classifier')
            
            print(f"≡ƒÄ» Training Model: {model_name}")
            print(f"≡ƒô¥ Reasoning: {recommended_model_info.get('reasoning', 'No reasoning provided')}")
            
            # Determine target variable (assume last column is target)
            target_column = df.columns[-1]
            
            # Apply column filtering if selected_columns is provided
            if selected_columns:
                print(f"\n🎯 APPLYING COLUMN FILTERING")
                print(f"📊 User selected {len(selected_columns)} columns for training")
                print(f"📊 Selected columns: {selected_columns}")
                
                # Filter out target column from selected columns if present
                feature_columns = [col for col in selected_columns if col != target_column]
                
                # Validate that all selected columns exist
                missing_columns = [col for col in feature_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Selected columns not found in dataset: {missing_columns}")
                
                X = df[feature_columns]
                print(f"✅ Filtered dataset to {len(X.columns)} selected feature columns")
                print(f"📊 Excluded {len(df.columns) - len(selected_columns)} columns from training")
            else:
                X = df.drop(columns=[target_column])
            
            y = df[target_column]
            
            print(f"≡ƒÄ» Target Variable: {target_column}")
            print(f"≡ƒôè Features: {list(X.columns)}")
            
            # Handle categorical variables (simple label encoding for now)
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = pd.Categorical(X[col]).codes
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"≡ƒôè Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
            
            # Determine problem type
            is_classification = user_data.get('data_type') == 'categorical' or len(y.unique()) < 20
            
            # Get the appropriate model
            model = self._get_model_instance(model_name, is_classification)
            
            print(f"≡ƒñû Model Type: {'Classification' if is_classification else 'Regression'}")
            print(f"≡ƒöº Model: {type(model).__name__}")
            
            # Train the model
            print(f"ΓÅ│ Training model...")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate performance metrics
            if is_classification:
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Γ£à Training Complete!")
                print(f"≡ƒÄ» Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                
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
                print(f"Γ£à Training Complete!")
                print(f"≡ƒôè RMSE: {rmse:.4f}")
                print(f"≡ƒôè MSE: {mse:.4f}")
                
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
            
            print(f"≡ƒÆ╛ Model saved to: {model_path}")
            print("="*80)
            
            return {
                'success': True,
                'model_name': model_name,
                'model_type': 'classification' if is_classification else 'regression',
                'performance': performance,
                'model_path': model_path,
                'feature_names': list(X.columns),
                'target_column': target_column,
                'selected_columns': selected_columns if selected_columns else [],
                'original_columns': list(df_original.columns),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_info': {
                    'feature_names': list(X.columns),
                    'target_column': target_column,
                    'selected_columns': selected_columns if selected_columns else [],
                    'original_columns': list(df_original.columns),
                    'excluded_columns': [col for col in df_original.columns if col not in (selected_columns or df_original.columns)]
                }
            }
            
        except Exception as e:
            print(f"Γ¥î Error training model: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def train_specific_model(self, file_path: str, model_name: str, user_data: Dict[str, Any], target_column: str = None, selected_columns: List[str] = None) -> Dict[str, Any]:
        """
        Train a specific model selected by the user with comprehensive preprocessing and detailed logging
        
        Args:
            file_path (str): Path to the dataset
            model_name (str): Specific model name to train
            user_data (dict): User-provided data about the problem
            target_column (str): Target column name (optional)
            selected_columns (List[str]): List of columns to use for training (optional, uses all if None)
            
        Returns:
            dict: Training results with model performance
        """
        try:
            import time
            from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
            from sklearn.impute import SimpleImputer
            
            print("\n" + "="*100)
            print(f"≡ƒÄ» TRAINING SPECIFIC MODEL: {model_name}")
            print("="*100)
            
            # Create model-specific directory
            model_folder = model_name.replace(' ', '_').replace('/', '_').lower()
            model_dir = os.path.join("models", model_folder)
            os.makedirs(model_dir, exist_ok=True)
            print(f"≡ƒôü Model directory created: {model_dir}")
            
            # ============================================================================
            # STEP 1: LOAD DATASET
            # ============================================================================
            print(f"\n{'='*80}")
            print("🔄 STEP 1: LOADING DATASET")
            print(f"{'='*80}")
            
            start_time = time.time()
            df_original = pd.read_csv(file_path)
            load_time = time.time() - start_time
            
            print(f"✅ Original dataset loaded successfully in {load_time:.2f} seconds")
            print(f"🔢 Original rows: {df_original.shape[0]}")
            print(f"🔢 Original columns: {df_original.shape[1]}")
            print(f"🔢 Memory usage: {df_original.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print(f"🔤 Original column names: {list(df_original.columns)}")
            
            # Apply column filtering FIRST if selected_columns is provided
            if selected_columns:
                print(f"\n🎯 APPLYING COLUMN FILTERING")
                print(f"📊 User selected {len(selected_columns)} columns for training")
                print(f"📊 Selected columns: {selected_columns}")
                
                # Validate that all selected columns exist in the dataset
                missing_columns = [col for col in selected_columns if col not in df_original.columns]
                if missing_columns:
                    raise ValueError(f"Selected columns not found in dataset: {missing_columns}")
                
                # Filter the entire dataframe to only include selected columns
                df = df_original[selected_columns]
                excluded_columns = [col for col in df_original.columns if col not in selected_columns]
                
                print(f"✅ Filtered dataset from {len(df_original.columns)} to {len(df.columns)} columns")
                print(f"📊 Training will use: {list(df.columns)}")
                print(f"🚫 Excluded from training: {excluded_columns}")
            else:
                df = df_original
                print(f"📊 No column filtering - using all {len(df.columns)} columns")
            
            print(f"🔢 Final dataset shape: {df.shape} (rows: {df.shape[0]}, columns: {df.shape[1]})")
            print(f"\n🔢 Final data types:")
            for col, dtype in df.dtypes.items():
                print(f"   - {col}: {dtype}")
            
            # ============================================================================
            # STEP 2: INITIAL DATA INSPECTION
            # ============================================================================
            print(f"\n{'='*80}")
            print("≡ƒöì STEP 2: INITIAL DATA INSPECTION")
            print(f"{'='*80}")
            
            print(f"\n≡ƒôè Missing values per column:")
            missing_counts = df.isnull().sum()
            for col, count in missing_counts.items():
                if count > 0:
                    percentage = (count / len(df)) * 100
                    print(f"   ΓÜá∩╕Å  {col}: {count} ({percentage:.2f}%)")
                else:
                    print(f"   Γ£à {col}: 0 (0.00%)")
            
            print(f"\n≡ƒôè Duplicate rows: {df.duplicated().sum()}")
            
            print(f"\n≡ƒôè Statistical summary:")
            print(df.describe())
            
            # ============================================================================
            # STEP 3: IDENTIFY TARGET AND FEATURES
            # ============================================================================
            print(f"\n{'='*80}")
            print("≡ƒÄ» STEP 3: IDENTIFYING TARGET AND FEATURES")
            print(f"{'='*80}")
            
            # Check if this is unlabeled data (clustering)
            # Handle both string and boolean values
            is_labeled_value = user_data.get('is_labeled', 'labeled')
            is_labeled = (
                is_labeled_value in ['labeled', 'true', True] or 
                (isinstance(is_labeled_value, bool) and is_labeled_value)
            )
            is_clustering = (
                is_labeled_value in ['unlabeled', 'false', False] or 
                (isinstance(is_labeled_value, bool) and not is_labeled_value) or
                not is_labeled
            )
            
            # Check if model is a clustering model
            clustering_models = ['kmeans', 'dbscan', 'hierarchical clustering', 'gaussian mixture', 'gmm']
            model_name_lower = model_name.lower()
            is_clustering_model = any(cluster_model in model_name_lower for cluster_model in clustering_models)
            
            # If unlabeled or clustering model, use all columns as features (excluding unique identifiers)
            if is_clustering or is_clustering_model:
                print(f"≡ƒÄ» Unlabeled data detected - using columns as features (clustering)")
                X = df.copy()
                
                # Apply column filtering if selected_columns is provided
                if selected_columns:
                    print(f"\n🎯 APPLYING COLUMN FILTERING FOR CLUSTERING")
                    print(f"📊 User selected {len(selected_columns)} columns for clustering")
                    print(f"📊 Selected columns: {selected_columns}")
                    
                    # Validate that all selected columns exist
                    missing_columns = [col for col in selected_columns if col not in X.columns]
                    if missing_columns:
                        raise ValueError(f"Selected columns not found in dataset: {missing_columns}")
                    
                    # Filter X to only include selected columns
                    X = X[selected_columns]
                    print(f"✅ Filtered dataset to {len(X.columns)} selected columns for clustering")
                else:
                    # Exclude unique identifier columns (ID, customerid, etc.)
                    id_patterns = ['id', 'customerid', 'userid', 'user_id', 'customer_id', 'index', 'idx']
                    columns_to_exclude = []
                    for col in X.columns:
                        col_lower = col.lower().strip()
                        # Check if column name matches ID patterns
                        if any(pattern in col_lower for pattern in id_patterns):
                            # Also check if it's a unique identifier (all values are unique or nearly unique)
                            unique_ratio = X[col].nunique() / len(X)
                            if unique_ratio > 0.95:  # More than 95% unique values
                                columns_to_exclude.append(col)
                                print(f"   ⚠️ Excluding unique identifier column: {col} ({unique_ratio*100:.1f}% unique values)")
                    
                    if columns_to_exclude:
                        X = X.drop(columns=columns_to_exclude)
                        print(f"   ✅ Excluded {len(columns_to_exclude)} unique identifier column(s)")
                
                y = None
                target_column = None
                print(f"\n🔍 Feature columns ({len(X.columns)}):")
                for i, col in enumerate(X.columns, 1):
                    print(f"   {i}. {col} ({X[col].dtype})")
            else:
                # Labeled data - need target column
                if target_column is None:
                    target_column = df.columns[-1]
                    print(f"Γä╣∩╕Å  No target column specified, using last column as default")
                
                # Validate target column exists
                if target_column not in df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}")
                
                print(f"≡ƒÄ» Target column: {target_column}")
                print(f"≡ƒôè Target data type: {df[target_column].dtype}")
                print(f"≡ƒôè Unique target values: {df[target_column].nunique()}")
                print(f"≡ƒôè Target value distribution:")
                print(df[target_column].value_counts().head(10))
                
                # Separate features and target
                X = df.drop(columns=[target_column])
                y = df[target_column]
                
                # Column filtering was already applied at the beginning of this function
                # X already contains only the selected columns (minus target)
                print(f"📊 Using pre-filtered feature columns: {list(X.columns)}")
                
                print(f"\n🔍 Feature columns ({len(X.columns)}):")
                for i, col in enumerate(X.columns, 1):
                    print(f"   {i}. {col} ({X[col].dtype})")
            
            # ============================================================================
            # STEP 4: DATA PREPROCESSING
            # ============================================================================
            print(f"\n{'='*80}")
            print("≡ƒöº STEP 4: DATA PREPROCESSING")
            print(f"{'='*80}")
            
            # 4.1: Handle missing values
            print(f"\n≡ƒöº Step 4.1: Handling Missing Values")
            print(f"{'-'*80}")
            
            # Identify numeric and categorical columns
            numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            print(f"≡ƒôè Numeric columns: {len(numeric_cols)}")
            for col in numeric_cols:
                print(f"   - {col}")
            print(f"≡ƒôè Categorical columns: {len(categorical_cols)}")
            for col in categorical_cols:
                print(f"   - {col}")
            
            # Handle missing values in numeric columns
            if numeric_cols:
                missing_numeric = X[numeric_cols].isnull().sum()
                if missing_numeric.any():
                    print(f"\n≡ƒöº Imputing missing numeric values with median...")
                    numeric_imputer = SimpleImputer(strategy='median')
                    X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
                    print(f"Γ£à Numeric columns imputed successfully")
                else:
                    print(f"Γ£à No missing values in numeric columns")
            
            # Handle missing values in categorical columns
            if categorical_cols:
                missing_categorical = X[categorical_cols].isnull().sum()
                if missing_categorical.any():
                    print(f"\n≡ƒöº Imputing missing categorical values with mode...")
                    categorical_imputer = SimpleImputer(strategy='most_frequent')
                    X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
                    print(f"Γ£à Categorical columns imputed successfully")
                else:
                    print(f"Γ£à No missing values in categorical columns")
            
            # 4.2: Handle duplicates
            print(f"\n≡ƒöº Step 4.2: Handling Duplicate Rows")
            print(f"{'-'*80}")
            original_rows = len(X)
            X = X.drop_duplicates()
            if y is not None:
                y = y[X.index]
            duplicates_removed = original_rows - len(X)
            if duplicates_removed > 0:
                print(f"≡ƒùæ∩╕Å  Removed {duplicates_removed} duplicate rows")
            else:
                print(f"Γ£à No duplicate rows found")
            
            # 4.3: Encode categorical variables
            print(f"\n≡ƒöº Step 4.3: Encoding Categorical Variables")
            print(f"{'-'*80}")
            
            label_encoders = {}
            if categorical_cols:
                for col in categorical_cols:
                    print(f"≡ƒöä Encoding column: {col}")
                    print(f"   Original unique values: {X[col].nunique()}")
                    print(f"   Sample values: {X[col].unique()[:5]}")
                    
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    label_encoders[col] = le
                    
                    print(f"   Γ£à Encoded to: {X[col].unique()[:5]}")
            else:
                print(f"Γä╣∩╕Å  No categorical columns to encode")
            
            # 4.4: Handle target encoding for classification (only for labeled data)
            print(f"\n≡ƒöº Step 4.4: Processing Target Variable")
            print(f"{'-'*80}")
            
            target_encoder = None
            is_classification = False
            
            if y is not None:
                is_labeled = user_data.get('is_labeled', 'labeled') in ['labeled', 'true', True]
                data_type = user_data.get('data_type', '')
                
                # Respect user's explicit choice for data type
                if data_type == 'continuous':
                    is_classification = False  # Force regression for continuous data
                elif data_type in ['categorical', 'classification']:
                    is_classification = True   # Force classification for categorical data
                else:
                    # Auto-detect only if no explicit choice
                    is_classification = y.nunique() < 20
                
                print(f"≡ƒöì Training mode detection:")
                print(f"   - is_labeled: {is_labeled} (value: {user_data.get('is_labeled')})")
                print(f"   - data_type: {data_type}")
                print(f"   - is_classification: {is_classification}")
                print(f"   - target unique values: {y.nunique()}")
                print(f"   - mode: {'Classification (respecting user choice)' if data_type in ['categorical', 'classification'] else 'Regression (respecting user choice)' if data_type == 'continuous' else 'Auto-detected'}")
                
                if is_classification and y.dtype == 'object':
                    print(f"≡ƒöä Encoding target variable (classification)")
                    print(f"   Original unique values: {y.nunique()}")
                    print(f"   Sample values: {y.unique()[:5]}")
                    
                    target_encoder = LabelEncoder()
                    y = target_encoder.fit_transform(y.astype(str))
                    
                    print(f"   Γ£à Encoded to: {np.unique(y)}")
            else:
                print(f"Γä╣∩╕Å  No target variable (clustering/unlabeled data)")
            
            # 4.5: Feature scaling
            print(f"\n≡ƒöº Step 4.5: Feature Scaling")
            print(f"{'-'*80}")
            
            if numeric_cols:
                print(f"≡ƒôè Original feature ranges:")
                for col in numeric_cols[:5]:  # Show first 5
                    if col in X.columns:
                        print(f"   {col}: [{X[col].min():.2f}, {X[col].max():.2f}]")
                
                print(f"\n≡ƒöä Applying StandardScaler to numeric features...")
                scaler = StandardScaler()
                X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
                
                print(f"≡ƒôè Scaled feature ranges:")
                for col in numeric_cols[:5]:  # Show first 5
                    if col in X.columns:
                        print(f"   {col}: [{X[col].min():.2f}, {X[col].max():.2f}]")
                print(f"Γ£à Features scaled successfully")
            else:
                scaler = None
                print(f"Γä╣∩╕Å  No numeric features to scale")
            
            # 4.6: Handle outliers
            print(f"\n≡ƒöº Step 4.6: Outlier Detection")
            print(f"{'-'*80}")
            
            if numeric_cols:
                for col in numeric_cols[:5]:  # Check first 5 numeric columns
                    if col in X.columns:
                        Q1 = X[col].quantile(0.25)
                        Q3 = X[col].quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = ((X[col] < (Q1 - 1.5 * IQR)) | (X[col] > (Q3 + 1.5 * IQR))).sum()
                        if outliers > 0:
                            percentage = (outliers / len(X)) * 100
                            print(f"   ΓÜá∩╕Å  {col}: {outliers} outliers ({percentage:.2f}%)")
                        else:
                            print(f"   Γ£à {col}: No outliers detected")
            
            # ============================================================================
            # STEP 5: TRAIN-TEST SPLIT (only for labeled data)
            # ============================================================================
            print(f"\n{'='*80}")
            print("Γ£é∩╕Å  STEP 5: SPLITTING DATA INTO TRAIN AND TEST SETS")
            print(f"{'='*80}")
            
            # For clustering/unlabeled data, skip train-test split
            if y is None or is_clustering or is_clustering_model:
                print(f"≡ƒöì Unlabeled/clustering data - using all data for training")
                X_train = X
                X_test = None
                y_train = None
                y_test = None
                print(f"≡ƒôè Training set size: {X_train.shape[0]} samples (100%)")
                print(f"≡ƒôè Feature dimensions: {X_train.shape[1]}")
            else:
                test_size = 0.2
                random_state = 42
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y if is_classification else None
                )
                
                print(f"≡ƒôè Training set size: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
                print(f"≡ƒôè Test set size: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
                print(f"≡ƒôè Feature dimensions: {X_train.shape[1]}")
                print(f"≡ƒôè Random state: {random_state}")
                if is_classification:
                    print(f"≡ƒôè Stratified split: Yes (maintains class distribution)")
                    print(f"\n≡ƒôè Training set class distribution:")
                    print(pd.Series(y_train).value_counts())
                    print(f"\n≡ƒôè Test set class distribution:")
                    print(pd.Series(y_test).value_counts())
            
            # ============================================================================
            # STEP 6: MODEL SELECTION AND CONFIGURATION
            # ============================================================================
            print(f"\n{'='*80}")
            print("≡ƒñû STEP 6: MODEL SELECTION AND CONFIGURATION")
            print(f"{'='*80}")
            
            print(f"≡ƒÄ» Selected model: {model_name}")
            print(f"≡ƒôè Problem type: {'Classification' if is_classification else 'Regression'}")
            print(f"≡ƒôè Labeled data: {'Yes' if is_labeled else 'No'}")
            print(f"≡ƒöì DEBUG: is_labeled value = {is_labeled}, type = {type(is_labeled)}")
            print(f"≡ƒöì DEBUG: user_data['is_labeled'] = {user_data.get('is_labeled')}")
            print(f"≡ƒöì DEBUG: model_name = {model_name}")
            
            # Get the specific model instance
            # For unlabeled data, ALWAYS use unsupervised learning
            # Also use unsupervised for clustering models even if labeled data is provided
            should_use_unsupervised = (
                is_clustering or  # Data is unlabeled
                is_clustering_model or  # Model is a clustering model
                y is None or  # No target variable
                not is_labeled or  # Explicitly marked as unlabeled
                'cluster' in model_name.lower() or  # Model name contains cluster
                'kmeans' in model_name.lower() or  # KMeans
                'dbscan' in model_name.lower() or  # DBSCAN
                'gmm' in model_name.lower() or  # Gaussian Mixture
                'gaussian mixture' in model_name.lower() or  # Gaussian Mixture
                'pca' in model_name.lower() or  # PCA
                'tsne' in model_name.lower() or  # t-SNE
                't-sne' in model_name.lower() or  # t-SNE
                'umap' in model_name.lower()  # UMAP
            )
            
            if should_use_unsupervised:
                print(f"≡ƒöì Unsupervised learning mode activated")
                print(f"≡ƒöì DEBUG: Reason - is_clustering={is_clustering}, is_clustering_model={is_clustering_model}, y is None={y is None}, not is_labeled={not is_labeled}")
                print(f"≡ƒöì DEBUG: user_data['is_labeled']={user_data.get('is_labeled')}, model_name={model_name}")
                # Store feature names before training (after ID exclusion)
                feature_names = list(X.columns)
                print(f"📊 Feature names to be used: {feature_names}")
                result = self._train_unsupervised_model(X, model_name, model_dir, selected_columns, df_original)
                return result
            else:
                print(f"≡ƒÜÇ SWITCHING TO REALISTIC COMPREHENSIVE TRAINING")
                print(f"{'='*80}")
                print(f"≡ƒöº Using advanced pipeline training with realistic timing (27-387 seconds)")
                print(f"≡ƒöº Model-specific parameter grids and comprehensive evaluation")
                
                # Use the realistic training method instead of the quick training
                mapped_model_name = self._map_model_name(model_name)
                print(f"≡ƒöä Mapped model name: '{model_name}' -> '{mapped_model_name}'")
                
                realistic_result = self._execute_pipeline_training(
                    model_name=mapped_model_name,
                    original_name=model_name,
                    file_path=file_path,
                    target_column=target_column,
                    selected_columns=selected_columns
                )
                
                if realistic_result['success']:
                    # Use the corrected metrics from the comprehensive training
                    ui_metrics = realistic_result.get('performance', {})
                    training_details = realistic_result.get('training_details', {})
                    
                    # Convert comprehensive realistic result to expected format
                    return {
                        'success': True,
                        'performance': ui_metrics,  # UI-friendly metrics from comprehensive training
                        'training_details': training_details,  # Training info from comprehensive training
                        'feature_info': realistic_result.get('feature_info', {}),  # Feature info from comprehensive training
                        'model_info': {
                            'name': realistic_result['model_name'],
                            'type': realistic_result.get('scenario', 'regression'),
                            'model_folder': realistic_result.get('model_folder', 'models/unknown'),
                            'model_directory': realistic_result.get('model_folder', 'models/unknown'),
                            'training_time': realistic_result.get('training_time', 30.0),
                            'test_score': realistic_result.get('test_score', 0.0),
                            'score_name': realistic_result.get('score_name', 'r2_score')
                        }
                    }
                else:
                    return realistic_result
            
            if is_classification:
                print(f"\n≡ƒôè Classification Metrics:")
                print(f"{'-'*80}")
                
                accuracy = accuracy_score(y_test, y_pred)
                print(f"≡ƒÄ» Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                
                print(f"≡ƒôè Precision (macro avg): {report['macro avg']['precision']:.4f}")
                print(f"≡ƒôè Recall (macro avg): {report['macro avg']['recall']:.4f}")
                print(f"≡ƒôè F1-score (macro avg): {report['macro avg']['f1-score']:.4f}")
                
                print(f"\n≡ƒôï Detailed Classification Report:")
                print(classification_report(y_test, y_pred, zero_division=0))
                
                performance = {
                    'model_name': model_name,
                    'model_type': 'classification',
                    'accuracy': float(accuracy),
                    'precision': float(report['macro avg']['precision']),
                    'recall': float(report['macro avg']['recall']),
                    'f1_score': float(report['macro avg']['f1-score']),
                    'training_time': float(training_time),
                    'prediction_time': float(prediction_time)
                }
                
            else:
                print(f"\n≡ƒôè Regression Metrics:")
                print(f"{'-'*80}")
                
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                print(f"≡ƒôè Mean Squared Error (MSE): {mse:.4f}")
                print(f"≡ƒôè Root Mean Squared Error (RMSE): {rmse:.4f}")
                print(f"≡ƒôè Mean Absolute Error (MAE): {mae:.4f}")
                print(f"≡ƒôè R┬▓ Score: {r2:.4f}")
                
                performance = {
                    'model_name': model_name,
                    'model_type': 'regression',
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2_score': float(r2),
                    'training_time': float(training_time),
                    'prediction_time': float(prediction_time)
                }
            
            # ============================================================================
            # STEP 9: MODEL PERSISTENCE
            # ============================================================================
            print(f"\n{'='*80}")
            print("≡ƒÆ╛ STEP 9: SAVING MODEL AND ARTIFACTS")
            print(f"{'='*80}")
            
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            
            # Save trained model
            model_filename = f"{model_folder}_{timestamp}.joblib"
            model_path = os.path.join(model_dir, model_filename)
            joblib.dump(model, model_path)
            print(f"Γ£à Model saved: {model_path}")
            
            # Save preprocessing artifacts
            if scaler is not None:
                scaler_path = os.path.join(model_dir, f"scaler_{timestamp}.joblib")
                joblib.dump(scaler, scaler_path)
                print(f"Γ£à Scaler saved: {scaler_path}")
            
            if label_encoders:
                encoders_path = os.path.join(model_dir, f"label_encoders_{timestamp}.joblib")
                joblib.dump(label_encoders, encoders_path)
                print(f"Γ£à Label encoders saved: {encoders_path}")
            
            if target_encoder is not None:
                target_encoder_path = os.path.join(model_dir, f"target_encoder_{timestamp}.joblib")
                joblib.dump(target_encoder, target_encoder_path)
                print(f"Γ£à Target encoder saved: {target_encoder_path}")
            
            # Save feature names
            feature_info = {
                'feature_names': list(X.columns),
                'numeric_features': numeric_cols,
                'categorical_features': categorical_cols,
                'target_column': target_column
            }
            feature_info_path = os.path.join(model_dir, f"feature_info_{timestamp}.json")
            with open(feature_info_path, 'w') as f:
                json.dump(feature_info, f, indent=2)
            print(f"Γ£à Feature info saved: {feature_info_path}")
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'model_type': 'classification' if is_classification else 'regression',
                'timestamp': timestamp,
                'training_samples': int(X_train.shape[0]),
                'test_samples': int(X_test.shape[0]),
                'num_features': int(X_train.shape[1]),
                'feature_names': list(X.columns),  # Add filtered feature names
                'target_column': target_column,     # Add target column
                'selected_columns': selected_columns if selected_columns else list(df_original.columns),  # Store original selection
                'performance': performance,
                'preprocessing': {
                    'scaler_used': scaler is not None,
                    'label_encoding_used': len(label_encoders) > 0,
                    'target_encoding_used': target_encoder is not None
                }
            }
            metadata_path = os.path.join(model_dir, f"metadata_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Γ£à Metadata saved: {metadata_path}")
            
            # ============================================================================
            # FINAL SUMMARY
            # ============================================================================
            print(f"\n{'='*100}")
            print(f"Γ£à MODEL TRAINING COMPLETED SUCCESSFULLY")
            print(f"{'='*100}")
            
            total_time = time.time() - start_time
            print(f"ΓÅ▒∩╕Å  Total execution time: {total_time:.2f} seconds")
            print(f"≡ƒôü Model directory: {model_dir}")
            print(f"≡ƒÄ» Model: {model_name}")
            print(f"≡ƒôè Performance summary: {performance}")
            print(f"{'='*100}\n")
            
            # Calculate detailed metrics for UI display
            # Prepare UI-friendly metrics based on model type
            if is_classification:
                ui_metrics = {
                    'accuracy': performance.get('accuracy', 0),
                    'precision': performance.get('precision', 0),
                    'recall': performance.get('recall', 0),
                    'f1_score': performance.get('f1_score', 0)
                }
            else:  # Regression
                ui_metrics = {
                    'r2_score': performance.get('r2_score', 0),
                    'mse': performance.get('mse', 0),
                    'rmse': performance.get('rmse', 0),
                    'mae': performance.get('mae', 0)
                }
            
            training_details = {
                'training_samples': int(X_train.shape[0]),
                'test_samples': int(X_test.shape[0]),
                'features': int(X_train.shape[1]),
                'training_time': total_time
            }
            
            return {
                'success': True,
                'performance': ui_metrics,  # UI-friendly metrics
                'training_details': training_details,  # Training info for UI
                'feature_info': {
                    'feature_names': list(X.columns),
                    'target_column': target_column,
                    'problem_type': 'classification' if is_classification else 'regression',
                    'selected_columns': selected_columns if selected_columns else [],
                    'original_columns': list(df_original.columns),
                    'excluded_columns': [col for col in df_original.columns if col not in (selected_columns or df_original.columns)],
                    'numeric_cols': numeric_cols,      # Add actual numeric columns from data
                    'categorical_cols': categorical_cols  # Add actual categorical columns from data
                },  # Feature information for frontend
                'model_info': {
                    'name': model_name,
                    'type': type(model).__name__,
                    'model_path': model_path,
                    'model_directory': model_dir,
                    'feature_count': int(X_train.shape[1]),
                    'training_samples': int(X_train.shape[0]),
                    'test_samples': int(X_test.shape[0]),
                    'feature_names': list(X.columns),  # Also add here for compatibility
                    'artifacts': {
                        'model': model_filename,
                        'scaler': f"scaler_{timestamp}.joblib" if scaler else None,
                        'label_encoders': f"label_encoders_{timestamp}.joblib" if label_encoders else None,
                        'target_encoder': f"target_encoder_{timestamp}.joblib" if target_encoder else None,
                        'feature_info': f"feature_info_{timestamp}.json",
                        'metadata': f"metadata_{timestamp}.json"
                    }
                }
            }
                
        except Exception as e:
            print(f"\n{'='*100}")
            print(f"Γ¥î ERROR IN MODEL TRAINING")
            print(f"{'='*100}")
            print(f"Error: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            print(f"{'='*100}\n")
            
            return {
                'success': False,
                'error': f"Training failed: {str(e)}",
                'model_name': model_name
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
    
    def _train_unsupervised_model(self, X: pd.DataFrame, model_name: str, model_dir: str, selected_columns: List[str] = None, df_original: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Train an unsupervised learning model with detailed logging
        
        Args:
            X (DataFrame): Feature dataset (already preprocessed, IDs excluded)
            model_name (str): Name of the unsupervised model
            model_dir (str): Directory to save the model
            selected_columns (List[str]): Originally selected columns by user
            df_original (DataFrame): Original dataframe before filtering
            
        Returns:
            dict: Training results with feature_names included
        """
        # Store feature names from the processed dataframe
        feature_names = list(X.columns)
        try:
            import time
            
            print(f"\n{'='*80}")
            print("≡ƒöì UNSUPERVISED LEARNING MODE")
            print(f"{'='*80}")
            
            # Get the unsupervised model instance
            model = self._get_unsupervised_model_instance(model_name)
            
            print(f"≡ƒöº Model class: {type(model).__name__}")
            print(f"≡ƒôï Model parameters:")
            params = model.get_params() if hasattr(model, 'get_params') else {}
            for param, value in params.items():
                print(f"   - {param}: {value}")
            
            print(f"\nΓÅ│ Training {model_name}...")
            print(f"≡ƒôè Training samples: {X.shape[0]}")
            print(f"≡ƒôè Features: {X.shape[1]}")
            
            # Preprocess data BEFORE training (clustering models need numeric data)
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.impute import SimpleImputer
            
            print(f"\n≡ƒöº Preprocessing data for clustering...")
            X_processed = X.copy()
            
            # Store preprocessing transformers for later use in prediction
            preprocessing_transformers = {
                'numeric_imputer': None,
                'scaler': None,
                'label_encoders': {}
            }
            
            # Handle missing values
            numeric_cols = X_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(numeric_cols) > 0:
                # Impute missing numeric values
                numeric_imputer = SimpleImputer(strategy='median')
                X_processed[numeric_cols] = numeric_imputer.fit_transform(X_processed[numeric_cols])
                preprocessing_transformers['numeric_imputer'] = numeric_imputer
                
                # Scale numeric columns
                scaler = StandardScaler()
                X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])
                preprocessing_transformers['scaler'] = scaler
                preprocessing_transformers['numeric_cols'] = numeric_cols
                print(f"   ✓ Processed {len(numeric_cols)} numeric columns")
            
            if len(categorical_cols) > 0:
                # Encode categorical columns
                for col in categorical_cols:
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                    preprocessing_transformers['label_encoders'][col] = le
                preprocessing_transformers['categorical_cols'] = categorical_cols
                print(f"   ✓ Encoded {len(categorical_cols)} categorical columns")
            
            # Convert to numpy array for sklearn models
            X_array = X_processed.values if hasattr(X_processed, 'values') else X_processed
            
            training_start = time.time()
            
            # Fit the model on preprocessed data
            if hasattr(model, 'fit_transform'):
                transformed_data = model.fit_transform(X_array)
                # Try to get labels from different attributes/methods
                if hasattr(model, 'labels_'):
                    labels = model.labels_
                elif hasattr(model, 'predict'):
                    labels = model.predict(X_array)
                else:
                    labels = None
            else:
                model.fit(X_array)
                transformed_data = None
                # Try to get labels from different attributes/methods
                if hasattr(model, 'labels_'):
                    labels = model.labels_
                elif hasattr(model, 'predict'):
                    labels = model.predict(X_array)
                else:
                    labels = None
            
            training_time = time.time() - training_start
            
            print(f"Γ£à Training completed in {training_time:.2f} seconds")
            
            # Use processed data for evaluation
            X_for_eval = X_array
            
            # Debug: Check if labels were obtained
            print(f"🔍 DEBUG: labels is None: {labels is None}")
            if labels is not None:
                print(f"🔍 DEBUG: labels shape: {labels.shape}, unique labels: {np.unique(labels) if hasattr(np, 'unique') else 'N/A'}")
            else:
                print(f"⚠️ WARNING: No labels obtained from model. Model type: {type(model).__name__}")
                print(f"   Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
            
            # Performance evaluation
            performance = {
                'model_name': model_name,
                'model_type': 'unsupervised',
                'training_time': float(training_time),
                'n_samples': int(X.shape[0]),
                'n_features': int(X.shape[1])
            }
            
            # Add model-specific metrics
            if labels is not None:
                unique_labels = np.unique(labels)
                n_clusters = len(unique_labels)
                
                print(f"\n≡ƒôè Clustering Results:")
                print(f"   Number of clusters: {n_clusters}")
                print(f"   Cluster distribution:")
                for label in unique_labels:
                    count = np.sum(labels == label)
                    percentage = (count / len(labels)) * 100
                    print(f"      Cluster {label}: {count} samples ({percentage:.2f}%)")
                
                performance['n_clusters'] = int(n_clusters)
                performance['cluster_distribution'] = {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
                
                # Calculate clustering quality metrics
                if n_clusters > 1 and n_clusters < len(X):
                    try:
                        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
                        
                        # X_for_eval is already a numpy array from preprocessing
                        # Silhouette Score (higher is better, range: -1 to 1)
                        silhouette_avg = silhouette_score(X_for_eval, labels)
                        performance['silhouette_score'] = float(silhouette_avg)
                        print(f"   Silhouette Score: {silhouette_avg:.4f}")
                        
                        # Davies-Bouldin Index (lower is better)
                        db_score = davies_bouldin_score(X_for_eval, labels)
                        performance['davies_bouldin_score'] = float(db_score)
                        print(f"   Davies-Bouldin Index: {db_score:.4f}")
                        
                        # Calinski-Harabasz Index (higher is better)
                        ch_score = calinski_harabasz_score(X_for_eval, labels)
                        performance['calinski_harabasz_score'] = float(ch_score)
                        print(f"   Calinski-Harabasz Index: {ch_score:.4f}")
                        
                        # For frontend compatibility, map silhouette score to a "main_score"
                        # Silhouette score ranges from -1 to 1, normalize to 0-1 for display
                        performance['main_score'] = float((silhouette_avg + 1) / 2)  # Normalize to 0-1
                        performance['score_name'] = 'silhouette_score'
                        
                    except Exception as metric_err:
                        print(f"⚠️ Could not calculate clustering metrics: {metric_err}")
                        performance['main_score'] = 0.5  # Default score
                        performance['score_name'] = 'n_clusters'
                else:
                    print(f"⚠️ Cannot calculate metrics: {n_clusters} clusters (need > 1 and < {len(X)})")
                    performance['main_score'] = 0.5
                    performance['score_name'] = 'n_clusters'
            
            if transformed_data is not None:
                print(f"\n≡ƒôè Transformed data shape: {transformed_data.shape}")
                performance['transformed_shape'] = list(transformed_data.shape)
            
            # Ensure main_score is set even if no labels
            if 'main_score' not in performance:
                performance['main_score'] = 0.5  # Default score
                performance['score_name'] = 'unsupervised'
            
            # Save the model
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            model_folder = model_name.replace(' ', '_').replace('/', '_').lower()
            model_filename = f"{model_folder}_{timestamp}.joblib"
            model_path = os.path.join(model_dir, model_filename)
            
            joblib.dump(model, model_path)
            print(f"\n≡ƒÆ╛ Model saved: {model_path}")
            
            # Save preprocessing transformers for prediction
            if preprocessing_transformers.get('scaler') is not None:
                scaler_path = os.path.join(model_dir, f"scaler_{timestamp}.joblib")
                joblib.dump(preprocessing_transformers['scaler'], scaler_path)
                print(f"≡ƒÆ╛ Scaler saved: {scaler_path}")
            
            if preprocessing_transformers.get('numeric_imputer') is not None:
                imputer_path = os.path.join(model_dir, f"numeric_imputer_{timestamp}.joblib")
                joblib.dump(preprocessing_transformers['numeric_imputer'], imputer_path)
                print(f"≡ƒÆ╛ Numeric imputer saved: {imputer_path}")
            
            if preprocessing_transformers.get('label_encoders'):
                encoders_path = os.path.join(model_dir, f"label_encoders_{timestamp}.joblib")
                joblib.dump(preprocessing_transformers['label_encoders'], encoders_path)
                print(f"≡ƒÆ╛ Label encoders saved: {encoders_path}")
            
            # Save feature info with column types
            feature_info = {
                'feature_names': feature_names,
                'numeric_cols': preprocessing_transformers.get('numeric_cols', []),
                'categorical_cols': preprocessing_transformers.get('categorical_cols', []),
                'target_column': None,
                'problem_type': 'unsupervised',
                'selected_columns': selected_columns if selected_columns else feature_names
            }
            feature_info_path = os.path.join(model_dir, f"feature_info_{timestamp}.json")
            with open(feature_info_path, 'w') as f:
                json.dump(feature_info, f, indent=2)
            print(f"≡ƒÆ╛ Feature info saved: {feature_info_path}")
            
            # Save transformed data if available
            if transformed_data is not None:
                transformed_path = os.path.join(model_dir, f"transformed_data_{timestamp}.npy")
                np.save(transformed_path, transformed_data)
                print(f"≡ƒÆ╛ Transformed data saved: {transformed_path}")
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'model_type': 'unsupervised',
                'timestamp': timestamp,
                'n_samples': int(X.shape[0]),
                'n_features': int(X.shape[1]),
                'performance': performance,
                'feature_names': feature_names,  # Include feature names for prediction
                'target_column': None,  # No target for unsupervised
                'selected_columns': selected_columns if selected_columns else feature_names,  # Store user selection or actual features used
                'numeric_cols': preprocessing_transformers.get('numeric_cols', []),
                'categorical_cols': preprocessing_transformers.get('categorical_cols', []),
                'preprocessing_artifacts': {
                    'scaler': f"scaler_{timestamp}.joblib" if preprocessing_transformers.get('scaler') else None,
                    'numeric_imputer': f"numeric_imputer_{timestamp}.joblib" if preprocessing_transformers.get('numeric_imputer') else None,
                    'label_encoders': f"label_encoders_{timestamp}.joblib" if preprocessing_transformers.get('label_encoders') else None,
                    'feature_info': f"feature_info_{timestamp}.json"
                }
            }
            metadata_path = os.path.join(model_dir, f"metadata_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"≡ƒÆ╛ Metadata saved: {metadata_path}")
            
            print(f"\n{'='*80}")
            print("Γ£à UNSUPERVISED MODEL TRAINING COMPLETED")
            print(f"{'='*80}\n")
            
            print(f"📊 Returning feature_names: {feature_names}")
            return {
                'success': True,
                'performance': performance,
                'model_info': {
                    'name': model_name,
                    'type': type(model).__name__,
                    'model_path': model_path,
                    'model_directory': model_dir,
                    'n_samples': int(X.shape[0]),
                    'n_features': int(X.shape[1]),
                    'feature_names': feature_names  # Include in model_info too
                },
                'feature_names': feature_names,  # Also at top level for easy access
                'training_details': {
                    'feature_names': feature_names,
                    'training_samples': int(X.shape[0]),
                    'features': int(X.shape[1])
                },
                'feature_info': {
                    'feature_names': feature_names,
                    'target_column': None,
                    'problem_type': 'unsupervised'
                }
            }
            
        except Exception as e:
            print(f"\nΓ¥î Error training unsupervised model: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            
            return {
                'success': False,
                'error': f"Unsupervised training failed: {str(e)}",
                'model_name': model_name
            }
    
    def _get_unsupervised_model_instance(self, model_name: str):
        """
        Get an instance of the specified unsupervised model
        
        Args:
            model_name (str): Name of the unsupervised model
            
        Returns:
            sklearn unsupervised model instance
        """
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        from sklearn.mixture import GaussianMixture
        from sklearn.decomposition import PCA
        
        model_name_lower = model_name.lower()
        
        # Clustering models
        if 'kmeans' in model_name_lower or 'k-means' in model_name_lower:
            return KMeans(n_clusters=3, random_state=42)
        elif 'dbscan' in model_name_lower:
            return DBSCAN(eps=0.5, min_samples=5)
        elif 'gaussian mixture' in model_name_lower or 'gmm' in model_name_lower or ('gaussian' in model_name_lower and 'mixture' in model_name_lower):
            return GaussianMixture(n_components=3, random_state=42)
        elif 'hierarchical' in model_name_lower or 'agglomerative' in model_name_lower:
            return AgglomerativeClustering(n_clusters=3)
        
        # Dimensionality reduction models
        elif 'pca' in model_name_lower or 'principal component' in model_name_lower:
            return PCA(n_components=2, random_state=42)
        elif 'tsne' in model_name_lower or 't-sne' in model_name_lower:
            from sklearn.manifold import TSNE
            return TSNE(n_components=2, random_state=42)
        elif 'umap' in model_name_lower:
            try:
                import umap
                return umap.UMAP(n_components=2, random_state=42)
            except ImportError:
                print("ΓÜá∩╕Å  UMAP not installed, falling back to PCA")
                return PCA(n_components=2, random_state=42)
        
        # Default to K-Means
        else:
            print(f"ΓÜá∩╕Å  Unknown unsupervised model '{model_name}', using K-Means as default")
            return KMeans(n_clusters=3, random_state=42)

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
    
    def generate_high_accuracy_training_script(self, model_name: str, file_path: str, target_column: str, 
                                                columns_to_drop: List[str] = None, 
                                                scoring_metric: str = None) -> Dict[str, Any]:
        """
        Generate a complete Python script using Pipeline + GridSearchCV for maximum accuracy
        
        Args:
            model_name (str): Name of the model to train (e.g., "Random Forest Classifier")
            file_path (str): Path to the dataset CSV file
            target_column (str): Name of the target column
            columns_to_drop (List[str]): List of columns to drop (IDs, unnecessary columns)
            scoring_metric (str): Scoring metric for GridSearchCV
            
        Returns:
            dict: Contains 'script' (Python code), 'model_info', and 'scenario_type'
        """
        try:
            # Load dataset to analyze
            df = pd.read_csv(file_path)
            
            # Determine scenario type
            if target_column and target_column in df.columns:
                # Labeled data
                target_series = df[target_column]
                
                # Check if classification or regression
                if target_series.dtype == 'object' or target_series.nunique() <= 20:
                    scenario_type = "classification"
                    default_scoring = "accuracy"
                else:
                    scenario_type = "regression"
                    default_scoring = "neg_mean_squared_error"
            else:
                # Unlabeled data (clustering)
                scenario_type = "clustering"
                default_scoring = "silhouette"
            
            # Use provided scoring or default
            final_scoring = scoring_metric if scoring_metric else default_scoring
            
            # Get model configuration
            model_config = self._get_model_configuration(model_name, scenario_type)
            
            if scenario_type in ["classification", "regression"]:
                script = self._generate_labeled_training_script(
                    file_path=file_path,
                    target_column=target_column,
                    columns_to_drop=columns_to_drop or [],
                    model_config=model_config,
                    scoring_metric=final_scoring,
                    scenario_type=scenario_type
                )
            else:
                script = self._generate_clustering_training_script(
                    file_path=file_path,
                    columns_to_drop=columns_to_drop or [],
                    model_config=model_config
                )
            
            return {
                'success': True,
                'script': script,
                'model_info': model_config,
                'scenario_type': scenario_type,
                'file_path': file_path,
                'target_column': target_column,
                'scoring_metric': final_scoring
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to generate training script: {str(e)}"
            }

    def _get_model_configuration(self, model_name: str, scenario_type: str) -> Dict[str, Any]:
        """Get model import, class, and parameter grid configuration"""
        
        # Model configurations with import statements and parameter grids
        model_configs = {
            # Classification Models
            "random forest classifier": {
                "import": "from sklearn.ensemble import RandomForestClassifier",
                "class": "RandomForestClassifier",
                "instance": "RandomForestClassifier(random_state=42)",
                "param_grid": {
                    "model__n_estimators": [100, 200, 300],
                    "model__max_depth": [5, 10, 15, None],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 4]
                }
            },
            "xgboost classifier": {
                "import": "from xgboost import XGBClassifier",
                "class": "XGBClassifier", 
                "instance": "XGBClassifier(random_state=42)",
                "param_grid": {
                    "model__n_estimators": [100, 200, 300],
                    "model__max_depth": [3, 6, 10],
                    "model__learning_rate": [0.01, 0.1, 0.2],
                    "model__subsample": [0.8, 0.9, 1.0]
                }
            },
            "lightgbm classifier": {
                "import": "from lightgbm import LGBMClassifier",
                "class": "LGBMClassifier",
                "instance": "LGBMClassifier(random_state=42, verbose=-1)",
                "param_grid": {
                    "model__n_estimators": [100, 200, 300],
                    "model__num_leaves": [31, 50, 100],
                    "model__learning_rate": [0.05, 0.1, 0.15],
                    "model__feature_fraction": [0.8, 0.9, 1.0]
                }
            },
            "catboost classifier": {
                "import": "from catboost import CatBoostClassifier",
                "class": "CatBoostClassifier",
                "instance": "CatBoostClassifier(random_state=42, verbose=False)",
                "param_grid": {
                    "model__iterations": [100, 200, 300],
                    "model__depth": [4, 6, 8],
                    "model__learning_rate": [0.05, 0.1, 0.15],
                    "model__l2_leaf_reg": [1, 3, 5]
                }
            },
            "support vector machines (svm)": {
                "import": "from sklearn.svm import SVC",
                "class": "SVC",
                "instance": "SVC(random_state=42, probability=True)",
                "param_grid": {
                    "model__C": [0.1, 1, 10, 100],
                    "model__kernel": ["linear", "rbf", "poly"],
                    "model__gamma": ["scale", "auto", 0.001, 0.01]
                }
            },
            "logistic regression": {
                "import": "from sklearn.linear_model import LogisticRegression",
                "class": "LogisticRegression",
                "instance": "LogisticRegression(random_state=42, max_iter=1000)",
                "param_grid": {
                    "model__C": [0.01, 0.1, 1, 10, 100],
                    "model__penalty": ["l1", "l2"],
                    "model__solver": ["liblinear", "saga"]
                }
            },
            
            # Regression Models  
            "random forest regressor": {
                "import": "from sklearn.ensemble import RandomForestRegressor",
                "class": "RandomForestRegressor",
                "instance": "RandomForestRegressor(random_state=42)",
                "param_grid": {
                    "model__n_estimators": [100, 200, 300],
                    "model__max_depth": [5, 10, 15, None],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 4]
                }
            },
            "xgboost regressor": {
                "import": "from xgboost import XGBRegressor",
                "class": "XGBRegressor",
                "instance": "XGBRegressor(random_state=42)",
                "param_grid": {
                    "model__n_estimators": [100, 200, 300],
                    "model__max_depth": [3, 6, 10],
                    "model__learning_rate": [0.01, 0.1, 0.2],
                    "model__subsample": [0.8, 0.9, 1.0]
                }
            },
            "lightgbm regressor": {
                "import": "from lightgbm import LGBMRegressor", 
                "class": "LGBMRegressor",
                "instance": "LGBMRegressor(random_state=42, verbose=-1)",
                "param_grid": {
                    "model__n_estimators": [100, 200, 300],
                    "model__num_leaves": [31, 50, 100],
                    "model__learning_rate": [0.05, 0.1, 0.15],
                    "model__feature_fraction": [0.8, 0.9, 1.0]
                }
            },
            
            # Clustering Models
            "kmeans": {
                "import": "from sklearn.cluster import KMeans",
                "class": "KMeans",
                "instance": "KMeans(n_init=10, random_state=42)",
                "param_grid": {
                    "model__n_clusters": [2, 3, 4, 5, 6, 7, 8]
                }
            },
            "dbscan": {
                "import": "from sklearn.cluster import DBSCAN",
                "class": "DBSCAN", 
                "instance": "DBSCAN()",
                "param_grid": {
                    "model__eps": [0.3, 0.5, 0.7, 1.0],
                    "model__min_samples": [3, 5, 7, 10]
                }
            }
        }
        
        # Find matching configuration
        model_key = model_name.lower().strip()
        return model_configs.get(model_key, {
            "import": "from sklearn.ensemble import RandomForestClassifier",
            "class": "RandomForestClassifier",
            "instance": "RandomForestClassifier(random_state=42)",
            "param_grid": {"model__n_estimators": [100, 200]}
        })

    def _generate_labeled_training_script(self, file_path: str, target_column: str, 
                                        columns_to_drop: List[str], model_config: Dict[str, Any],
                                        scoring_metric: str, scenario_type: str) -> str:
        """Generate complete Pipeline + GridSearchCV script for labeled data"""
        
        # Get filename from path
        filename = file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
        
        # Format columns to drop
        drop_cols_str = str([target_column] + columns_to_drop) if columns_to_drop else f"['{target_column}']"
        
        # Format parameter grid
        param_grid_str = str(model_config["param_grid"]).replace("'", "'")
        
        # Choose evaluation method
        if scenario_type == "classification":
            evaluation_import = "from sklearn.metrics import classification_report"
            evaluation_code = "print('\\nTest Set Evaluation:')\nprint(classification_report(y_test, y_pred))"
        else:
            evaluation_import = "from sklearn.metrics import mean_squared_error, r2_score"
            evaluation_code = "mse = mean_squared_error(y_test, y_pred)\nr2 = r2_score(y_test, y_pred)\nprint(f'\\nTest Set Evaluation:')\nprint(f'Mean Squared Error: {mse:.4f}')\nprint(f'R┬▓ Score: {r2:.4f}')"

        script = f'''"""
High-Accuracy Model Training Script
Generated using Pipeline + GridSearchCV for optimal performance
Model: {model_config["class"]}
Scenario: {scenario_type.title()}
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
{evaluation_import}
{model_config["import"]}

# Load Data
print("≡ƒôä Loading dataset...")
df = pd.read_csv('{filename}')
print(f"Dataset shape: {{df.shape}}")

# Define Target & Features
print("\\n≡ƒÄ» Defining target and features...")
target_column = '{target_column}'
columns_to_drop = {drop_cols_str}

# The target variable y is the column '{target_column}'
y = df[target_column]

# The features X are all columns except {drop_cols_str}
feature_columns = [col for col in df.columns if col not in columns_to_drop]
X = df[feature_columns]

print(f"Target variable: {{target_column}}")
print(f"Number of features: {{len(feature_columns)}}")
print(f"Features: {{feature_columns}}")

# Automatic Preprocessing
print("\\n≡ƒöä Setting up preprocessing pipeline...")

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numerical features: {{numeric_features}}")
print(f"Categorical features: {{categorical_features}}")

# Create preprocessing pipelines
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Create Full Pipeline
print("\\n≡ƒñû Creating full training pipeline...")
model = {model_config["instance"]}

# Create pipeline that chains preprocessing and model
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Data Split
print("\\n≡ƒôè Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if '{scenario_type}' == 'classification' else None
)

print(f"Training set: {{X_train.shape}}")
print(f"Test set: {{X_test.shape}}")

# Hyperparameter Tuning
print("\\n≡ƒöì Starting hyperparameter tuning...")
param_grid = {param_grid_str}

print(f"Parameter grid: {{param_grid}}")
print("Using 5-fold cross-validation...")

# Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=full_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='{scoring_metric}',
    n_jobs=-1,
    verbose=1
)

# Train & Evaluate
print("\\n≡ƒÜÇ Training model with grid search...")
grid_search.fit(X_train, y_train)

print("\\n≡ƒÅå TRAINING COMPLETED!")
print("=" * 50)

# Best parameters
print("Best Tuned Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {{param}}: {{value}}")

# Best cross-validation score
print(f"\\nBest Cross-Validation Score: {{grid_search.best_score_:.4f}}")

# Test set evaluation
print("\\n≡ƒôê Evaluating on test set...")
y_pred = grid_search.predict(X_test)

{evaluation_code}

# Feature importance (if available)
try:
    if hasattr(grid_search.best_estimator_.named_steps['model'], 'feature_importances_'):
        feature_names = (numeric_features + 
                        list(grid_search.best_estimator_.named_steps['preprocessor']
                             .named_transformers_['cat']
                             .named_steps['encoder']
                             .get_feature_names_out(categorical_features)))
        
        importances = grid_search.best_estimator_.named_steps['model'].feature_importances_
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("\\n≡ƒôè Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"  {{i+1:2d}}. {{feature:30s}} {{importance:.4f}}")
except:
    print("\\nFeature importance not available for this model.")

print("\\nΓ£à Training completed successfully!")
'''
        
        return script

    def _generate_clustering_training_script(self, file_path: str, columns_to_drop: List[str], 
                                           model_config: Dict[str, Any]) -> str:
        """Generate complete clustering script for unlabeled data"""
        
        filename = file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
        drop_cols_str = str(columns_to_drop) if columns_to_drop else "[]"
        
        script = f'''"""
High-Quality Clustering Script
Generated using Pipeline for optimal preprocessing
Model: {model_config["class"]}
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
{model_config["import"]}

# Load Data
print("≡ƒôä Loading dataset...")
df = pd.read_csv('{filename}')
print(f"Dataset shape: {{df.shape}}")

# Define Features
print("\\n≡ƒÄ» Defining features...")
columns_to_drop = {drop_cols_str}

# The features X are all columns except {drop_cols_str}
feature_columns = [col for col in df.columns if col not in columns_to_drop]
X = df[feature_columns]

print(f"Number of features: {{len(feature_columns)}}")
print(f"Features: {{feature_columns}}")

# Automatic Preprocessing
print("\\n≡ƒöä Setting up preprocessing pipeline...")

# Identify numerical and categorical columns  
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numerical features: {{numeric_features}}")
print(f"Categorical features: {{categorical_features}}")

# Create preprocessing pipelines
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Create Full Pipeline
print("\\n≡ƒñû Creating clustering pipeline...")
model = {model_config["instance"]}

# Create pipeline that chains preprocessing and clustering
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Train
print("\\n≡ƒÜÇ Training clustering model...")
cluster_labels = full_pipeline.fit_predict(X)

print(f"Γ£à Clustering completed!")
print(f"Number of clusters found: {{len(np.unique(cluster_labels))}}")

# Evaluate
print("\\n≡ƒôè Evaluating cluster quality...")

# Get preprocessed data for evaluation
X_preprocessed = full_pipeline.named_steps['preprocessor'].transform(X)

# Calculate Silhouette Score
if len(np.unique(cluster_labels)) > 1:
    silhouette_avg = silhouette_score(X_preprocessed, cluster_labels)
    print(f"Silhouette Score: {{silhouette_avg:.4f}}")
    
    if silhouette_avg > 0.5:
        print("≡ƒÅå Excellent clustering quality!")
    elif silhouette_avg > 0.25:
        print("Γ£à Good clustering quality")
    else:
        print("ΓÜá∩╕Å Clustering quality could be improved")
else:
    print("ΓÜá∩╕Å Only one cluster found - consider adjusting parameters")

# Cluster distribution
unique, counts = np.unique(cluster_labels, return_counts=True)
print("\\n≡ƒôê Cluster Distribution:")
for cluster_id, count in zip(unique, counts):
    percentage = (count / len(cluster_labels)) * 100
    print(f"  Cluster {{cluster_id}}: {{count}} points ({{percentage:.1f}}%)")

# Bonus: Elbow Method (for KMeans)
if 'kmeans' in model_config["class"].lower():
    print("\\n≡ƒöì Finding optimal number of clusters (Elbow Method)...")
    
    inertias = []
    k_range = range(1, min(11, len(X)))
    
    for k in k_range:
        kmeans_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', KMeans(n_clusters=k, n_init=10, random_state=42))
        ])
        kmeans_pipeline.fit(X)
        inertias.append(kmeans_pipeline.named_steps['model'].inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()
    
    print("Look for the 'elbow' in the curve to determine optimal k!")

print("\\nΓ£à Clustering analysis completed!")
'''
        
        return script

    def _map_model_name(self, recommendation_model_name: str) -> str:
        """
        Map model names from recommendation system to advanced trainer model names
        
        Args:
            recommendation_model_name (str): Model name from recommendation system
            
        Returns:
            str: Mapped model name for advanced trainer
        """
        # Model name mapping dictionary
        name_mapping = {
            # Classification models - Frontend naming convention
            "random-forest-classifier": "Random Forest",
            "Random Forest Classifier": "Random Forest",
            "xgboost-classifier": "XGBoost",
            "XGBoost Classifier": "XGBoost", 
            "lightgbm-classifier": "LightGBM",
            "LightGBM Classifier": "LightGBM",
            "catboost-classifier": "CatBoost",
            "CatBoost Classifier": "CatBoost",
            "svm-classifier": "Support Vector Machine",
            "Support Vector Machines (SVM)": "Support Vector Machine",
            "logistic-regression": "Logistic Regression",
            "Logistic Regression": "Logistic Regression",
            "neural-network-classifier": "Neural Network",
            "Neural Networks (MLP Classifier)": "Neural Network",
            "knn-classifier": "K-Neighbors",
            "K-Nearest Neighbors (KNN) Classifier": "K-Neighbors", 
            "decision-tree-classifier": "Decision Tree",
            "Decision Tree Classifier": "Decision Tree",
            "gradient-boosting-classifier": "Gradient Boosting",
            "Gradient Boosting Classifier": "Gradient Boosting",
            "naive-bayes": "Naive Bayes",
            "Naive Bayes": "Naive Bayes",
            
            # Regression models - Frontend naming convention
            "random-forest-regressor": "Random Forest Regressor",
            "Random Forest Regressor": "Random Forest Regressor",
            "xgboost-regressor": "XGBoost Regressor",
            "XGBoost Regressor": "XGBoost Regressor",
            "lightgbm-regressor": "LightGBM Regressor",
            "LightGBM Regressor": "LightGBM Regressor", 
            "catboost-regressor": "CatBoost Regressor",
            "CatBoost Regressor": "CatBoost Regressor",
            "svm-regressor": "Support Vector Regressor",
            "Support Vector Regressor": "Support Vector Regressor",
            "linear-regression": "Linear Regression",
            "Linear Regression": "Linear Regression",
            "ridge-regression": "Ridge Regression",
            "Ridge Regression": "Ridge Regression",
            "lasso-regression": "Lasso Regression",
            "Lasso Regression": "Lasso Regression",
            "elastic-net": "ElasticNet",
            "Elastic Net": "ElasticNet",
            "gradient-boosting-regressor": "Gradient Boosting Regressor",
            "Gradient Boosting Regressor": "Gradient Boosting Regressor",
            "neural-network-regressor": "Neural Network Regressor",
            "Neural Networks (MLP Regressor)": "Neural Network Regressor",
            
            # Clustering models
            "KMeans": "KMeans",
            "DBSCAN": "DBSCAN",
            "Hierarchical Clustering": "Hierarchical Clustering"
        }
        
        # Check for exact match first
        if recommendation_model_name in name_mapping:
            return name_mapping[recommendation_model_name]
        
        # If no exact match, try to find partial match
        recommendation_lower = recommendation_model_name.lower()
        for rec_name, trainer_name in name_mapping.items():
            if rec_name.lower() == recommendation_lower:
                return trainer_name
        
        # If still no match, try smart matching based on keywords
        if "random" in recommendation_lower and "forest" in recommendation_lower:
            return "Random Forest Regressor" if "regressor" in recommendation_lower else "Random Forest"
        elif "xgboost" in recommendation_lower:
            return "XGBoost Regressor" if "regressor" in recommendation_lower else "XGBoost"
        elif "lightgbm" in recommendation_lower:
            return "LightGBM Regressor" if "regressor" in recommendation_lower else "LightGBM"
        elif "svm" in recommendation_lower or "support vector" in recommendation_lower:
            return "Support Vector Regressor" if "regressor" in recommendation_lower else "Support Vector Machine"
        elif "logistic" in recommendation_lower:
            return "Logistic Regression"
        elif "neural" in recommendation_lower or "mlp" in recommendation_lower:
            return "Neural Network Regressor" if "regressor" in recommendation_lower else "Neural Network"
        elif "decision" in recommendation_lower and "tree" in recommendation_lower:
            return "Decision Tree Regressor" if "regressor" in recommendation_lower else "Decision Tree"
        elif "gradient" in recommendation_lower and "boost" in recommendation_lower:
            return "Gradient Boosting Regressor" if "regressor" in recommendation_lower else "Gradient Boosting"
        elif "naive" in recommendation_lower and "bayes" in recommendation_lower:
            return "Naive Bayes"
        elif "knn" in recommendation_lower or "neighbor" in recommendation_lower:
            return "K-Neighbors Regressor" if "regressor" in recommendation_lower else "K-Neighbors"
            return "LightGBM Regressor" if "regressor" in recommendation_lower else "LightGBM"
        elif "svm" in recommendation_lower or "support vector" in recommendation_lower:
            return "Support Vector Regressor" if "regressor" in recommendation_lower else "Support Vector Machine"
        elif "logistic" in recommendation_lower:
            return "Logistic Regression"
        elif "neural" in recommendation_lower or "mlp" in recommendation_lower:
            return "Neural Network Regressor" if "regressor" in recommendation_lower else "Neural Network"
        elif "knn" in recommendation_lower or "k-nearest" in recommendation_lower:
            return "K-Neighbors"
        elif "decision tree" in recommendation_lower:
            return "Decision Tree"
        elif "gradient boost" in recommendation_lower:
            return "Gradient Boosting Regressor" if "regressor" in recommendation_lower else "Gradient Boosting"
        elif "naive bayes" in recommendation_lower:
            return "Naive Bayes"
        elif "kmeans" in recommendation_lower:
            return "KMeans"
        elif "dbscan" in recommendation_lower:
            return "DBSCAN"
        
        # If no mapping found, return original name and let trainer handle error
        print(f"ΓÜá∩╕Å  No mapping found for model '{recommendation_model_name}', using original name")
        return recommendation_model_name

    def _execute_pipeline_training(self, model_name: str, original_name: str, file_path: str, target_column: str = None, selected_columns: List[str] = None) -> Dict[str, Any]:
        """
        Execute comprehensive model-specific training with realistic timing and high accuracy
        
        This method implements:
        - Model-specific parameter grids for each algorithm
        - Comprehensive preprocessing pipeline 
        - Realistic training times (30-120 seconds depending on model complexity)
        - Detailed progress logging with step-by-step tracking
        - High accuracy targeting with intelligent retraining
        
        NOTE: This method is ONLY for labeled (supervised) data. For unlabeled data, use _train_unsupervised_model instead.
        """
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
        from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
        from sklearn.model_selection import cross_val_score
        import os
        import time
        from datetime import datetime
        import joblib
        
        try:
            # Safety check: This method should NOT be called for unlabeled data
            if target_column is None:
                raise ValueError("❌ _execute_pipeline_training called without target_column. This method is only for labeled data. Use _train_unsupervised_model for unlabeled data.")
            
            print(f"\n{'='*100}")
            print(f"≡ƒÜÇ STARTING COMPREHENSIVE TRAINING FOR: {model_name.upper()}")
            print(f"{'='*100}")
            
            # 1. Data Loading and Analysis
            print(f"\n≡ƒôä STEP 1: LOADING AND ANALYZING DATA")
            print(f"{'='*60}")
            
            start_time = time.time()
            df = pd.read_csv(file_path)
            loading_time = time.time() - start_time
            
            print(f"Γ£à Dataset loaded successfully in {loading_time:.2f} seconds")
            print(f"≡ƒôè Dataset shape: {df.shape[0]} rows ├ù {df.shape[1]} columns")
            print(f"≡ƒôè Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Validate target column
            if target_column not in df.columns:
                raise ValueError(f"Γ¥î Target column '{target_column}' not found in dataset")
            
            # Data quality analysis
            print(f"\n≡ƒöì DATA QUALITY ANALYSIS:")
            missing_total = df.isnull().sum().sum()
            duplicate_total = df.duplicated().sum()
            print(f"   ≡ƒôè Missing values: {missing_total} ({missing_total/len(df)*100:.2f}%)")
            print(f"   ≡ƒôè Duplicate rows: {duplicate_total} ({duplicate_total/len(df)*100:.2f}%)")
            
            # Determine problem type
            unique_targets = df[target_column].nunique()
            target_dtype = df[target_column].dtype
            
            is_classification = (
                target_dtype in ['object', 'bool', 'category'] or
                (target_dtype in ['int64', 'int32'] and unique_targets <= 50) or
                (unique_targets <= 20 and len(df) > 100)
            )
            
            scenario = "classification" if is_classification else "regression"
            
            print(f"\n≡ƒÄ» PROBLEM TYPE ANALYSIS:")
            print(f"   ≡ƒôï Problem type: {scenario.upper()}")
            print(f"   ≡ƒÄ» Target column: '{target_column}'")
            print(f"   ≡ƒôè Target data type: {target_dtype}")
            print(f"   ≡ƒôè Unique target values: {unique_targets}")
            
            if scenario == "classification":
                target_dist = df[target_column].value_counts().head()
                print(f"   ≡ƒôè Class distribution:")
                for value, count in target_dist.items():
                    print(f"      ΓÇó {value}: {count} ({count/len(df)*100:.1f}%)")
            
            # 2. Feature Engineering and Preparation
            print(f"\n≡ƒöº STEP 2: COMPREHENSIVE DATA PREPROCESSING")
            print(f"{'='*60}")
            
            # Separate features and target
            y = df[target_column]
            X = df.drop(columns=[target_column])
            
            # Apply column filtering if selected_columns is provided
            if selected_columns:
                print(f"\n🎯 APPLYING COLUMN FILTERING FOR SUPERVISED LEARNING")
                print(f"📊 User selected {len(selected_columns)} columns for training")
                print(f"📊 Selected columns: {selected_columns}")
                
                # Filter to only selected columns (excluding target which is already separated)
                available_features = list(X.columns)
                selected_features = [col for col in selected_columns if col in available_features and col != target_column]
                
                if selected_features:
                    X = X[selected_features]
                    print(f"✅ Filtered dataset from {len(available_features)} to {len(selected_features)} features")
                    print(f"📊 Training will use: {list(X.columns)}")
                    print(f"🚫 Excluded from training: {[col for col in available_features if col not in selected_features]}")
                else:
                    print(f"⚠️ Warning: No valid features found in selected columns, using all features")
                    selected_columns = available_features  # Update selected_columns for metadata
            
            print(f"🔢 Final feature dataset shape: ({X.shape[0]} rows, {X.shape[1]} columns)")
            
            # Store original feature names for later use
            original_feature_names = list(X.columns)
            print(f"   ≡ƒôè Original feature names: {original_feature_names}")
            
            # Feature type identification
            numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            print(f"   ≡ƒôê Numerical features ({len(numeric_features)}): {numeric_features[:5]}{'...' if len(numeric_features) > 5 else ''}")
            print(f"   ≡ƒÅ╖∩╕Å  Categorical features ({len(categorical_features)}): {categorical_features[:5]}{'...' if len(categorical_features) > 5 else ''}")
            
            # Data preprocessing with pipeline
            preprocessor_steps = []
            
            if numeric_features:
                numeric_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
                preprocessor_steps.append(('num', numeric_pipeline, numeric_features))
                print(f"   Γ£à Numeric preprocessing: median imputation + standard scaling")
            
            if categorical_features:
                categorical_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
                ])
                preprocessor_steps.append(('cat', categorical_pipeline, categorical_features))
                print(f"   Γ£à Categorical preprocessing: mode imputation + one-hot encoding")
            
            if not preprocessor_steps:
                raise ValueError("Γ¥î No valid features found for preprocessing")
            
            preprocessor = ColumnTransformer(preprocessor_steps)
            
            # Handle target encoding
            target_encoder = None
            if scenario == "classification" and y.dtype == 'object':
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y)
                print(f"   Γ£à Target encoded from categorical to numeric")
            
            # 3. Train-Test Split
            print(f"\n≡ƒôè STEP 3: DATA SPLITTING")
            print(f"{'='*60}")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if scenario == "classification" and len(np.unique(y)) > 1 else None
            )
            
            print(f"   ≡ƒôè Training set: {X_train.shape[0]} samples ({80}%)")
            print(f"   ≡ƒôè Test set: {X_test.shape[0]} samples ({20}%)")
            print(f"   ≡ƒôè Feature dimensions: {X_train.shape[1]}")
            
            if scenario == "classification":
                print(f"   Γ£à Stratified split applied to maintain class balance")
            
            # 4. Model Selection and Configuration
            print(f"\n≡ƒñû STEP 4: MODEL CONFIGURATION")
            print(f"{'='*60}")
            
            model, param_grid = self._get_model_and_params(model_name, scenario)
            
            print(f"   ≡ƒÄ» Model: {type(model).__name__}")
            print(f"   ≡ƒöº Model parameters: {model.get_params() if hasattr(model, 'get_params') else 'N/A'}")
            print(f"   ≡ƒöì Parameter grid size: {len(param_grid)} parameters")
            
            # Show expected training time
            expected_times = {
                'random forest': '30-60 seconds',
                'xgboost': '45-90 seconds', 
                'lightgbm': '25-50 seconds',
                'catboost': '40-80 seconds',
                'svm': '60-120 seconds',
                'neural': '45-100 seconds',
                'gradient': '35-70 seconds',
                'logistic': '15-30 seconds',
                'naive': '10-25 seconds',
                'knn': '20-40 seconds',
                'decision': '15-35 seconds'
            }
            
            expected_time = next((time_range for key, time_range in expected_times.items() if key in model_name.lower()), '30-60 seconds')
            print(f"   ΓÅ▒∩╕Å  Expected training time: {expected_time}")
            
            # 5. Pipeline Creation
            print(f"\n≡ƒöº STEP 5: CREATING TRAINING PIPELINE")
            print(f"{'='*60}")
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            print(f"   Γ£à Pipeline created with {len(pipeline.steps)} steps:")
            for i, (name, step) in enumerate(pipeline.steps, 1):
                print(f"      {i}. {name}: {type(step).__name__}")
            
            # 6. Hyperparameter Tuning
            print(f"\n≡ƒöì STEP 6: HYPERPARAMETER OPTIMIZATION")
            print(f"{'='*60}")
            
            scoring = 'accuracy' if scenario == "classification" else 'r2'
            
            # Calculate parameter combinations for progress tracking
            total_combinations = 1
            for param_values in param_grid.values():
                if isinstance(param_values, list):
                    total_combinations *= len(param_values)
            
            # Determine appropriate CV folds based on dataset size
            n_samples = len(X_train)
            if scenario == 'classification':
                # For classification, ensure we have enough samples per class for CV
                min_class_size = min(np.bincount(y_train))
                cv_folds = min(5, min_class_size, n_samples // 2)
            else:
                # For regression, just ensure we have enough samples
                cv_folds = min(5, n_samples // 2)
            
            # Minimum of 2 folds
            cv_folds = max(2, cv_folds)
            
            print(f"   ≡ƒöì Using GridSearchCV with {cv_folds}-fold cross-validation")
            print(f"   ≡ƒôè Scoring metric: {scoring}")
            print(f"   ≡ƒöó Total parameter combinations: {total_combinations}")
            print(f"   ≡ƒöä Total model fits: {total_combinations * cv_folds} (CV folds)")
            
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                return_train_score=True
            )
            
            # 7. Model Training
            print(f"\n≡ƒÜÇ STEP 7: MODEL TRAINING & EVALUATION")
            print(f"{'='*60}")
            
            training_start = time.time()
            print(f"   ΓÅ│ Training started at {datetime.now().strftime('%H:%M:%S')}")
            print(f"   ≡ƒöä Training {total_combinations} configurations with {cv_folds}-fold CV...")
            
            # Fit the model with progress tracking
            grid_search.fit(X_train, y_train)
            training_time = time.time() - training_start
            
            print(f"   Γ£à Training completed in {training_time:.2f} seconds")
            print(f"   ≡ƒÅå Best CV score: {grid_search.best_score_:.4f}")
            
            print(f"\n   ≡ƒöº Best hyperparameters:")
            for param, value in grid_search.best_params_.items():
                print(f"      ΓÇó {param}: {value}")
            
            # 8. Model Evaluation
            print(f"\n≡ƒôê STEP 8: COMPREHENSIVE MODEL EVALUATION")
            print(f"{'='*60}")
            
            # Make predictions
            y_pred = grid_search.predict(X_test)
            
            if scenario == "classification":
                test_accuracy = accuracy_score(y_test, y_pred)
                print(f"   ≡ƒÄ» Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
                
                print(f"\n   ≡ƒôï Detailed Classification Report:")
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                print(classification_report(y_test, y_pred, zero_division=0))
                
                # Performance summary
                performance_metrics = {
                    'accuracy': float(test_accuracy),
                    'precision': float(report['macro avg']['precision']),
                    'recall': float(report['macro avg']['recall']),
                    'f1_score': float(report['macro avg']['f1-score'])
                }
                
            else:
                from sklearn.metrics import mean_absolute_error
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                print(f"   ≡ƒôè Test R┬▓ Score: {r2:.4f} ({r2*100:.2f}%)")
                print(f"   ≡ƒôè Root Mean Squared Error: {rmse:.4f}")
                print(f"   ≡ƒôè Mean Absolute Error: {mae:.4f}")
                
                performance_metrics = {
                    'r2_score': float(r2),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'mse': float(mse)
                }
            
            # 9. Model Persistence
            print(f"\n≡ƒÆ╛ STEP 9: SAVING MODEL AND ARTIFACTS")
            print(f"{'='*60}")
            
            # Create model directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_model_name = original_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            model_folder = f"models/{safe_model_name}"
            
            os.makedirs(model_folder, exist_ok=True)
            
            # Save the trained model
            model_path = os.path.join(model_folder, f"model_{timestamp}.joblib")
            joblib.dump(grid_search.best_estimator_, model_path)
            print(f"   Γ£à Model saved: {model_path}")
            
            # Save target encoder if used
            if target_encoder is not None:
                encoder_path = os.path.join(model_folder, f"target_encoder_{timestamp}.joblib")
                joblib.dump(target_encoder, encoder_path)
                print(f"   Γ£à Target encoder saved: {encoder_path}")
            
            # Save metadata
            metadata = {
                'model_name': original_name,
                'model_type': scenario,
                'timestamp': timestamp,
                'training_time': training_time,
                'best_cv_score': float(grid_search.best_score_),
                'test_performance': performance_metrics,
                'best_params': grid_search.best_params_,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'feature_names': original_feature_names,  # Use original feature names (filtered)
                'target_column': target_column,
                'selected_columns': selected_columns if selected_columns else original_feature_names,  # Store user selection
                'preprocessing_steps': len(preprocessor_steps)
            }
            
            metadata_path = os.path.join(model_folder, f"metadata_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            print(f"   Γ£à Metadata saved: {metadata_path}")
            
            # Final Summary
            print(f"\n{'='*100}")
            print(f"≡ƒÄë TRAINING COMPLETED SUCCESSFULLY FOR {original_name.upper()}")
            print(f"{'='*100}")
            print(f"≡ƒôü Model folder: {model_folder}")
            print(f"ΓÅ▒∩╕Å  Training time: {training_time:.2f} seconds")
            print(f"≡ƒÄ» Best CV score: {grid_search.best_score_:.4f}")
            if scenario == "classification":
                print(f"≡ƒÄ» Test accuracy: {performance_metrics['accuracy']:.4f}")
            else:
                print(f"≡ƒÄ» Test R┬▓ score: {performance_metrics['r2_score']:.4f}")
            print(f"≡ƒôè Total samples: {len(X)}")
            print(f"≡ƒôè Features used: {X.shape[1]}")
            print(f"{'='*100}\n")
            
            # Calculate detailed metrics for UI display
            if scenario == "classification":
                ui_metrics = {
                    'accuracy': performance_metrics.get('accuracy', 0),
                    'precision': performance_metrics.get('precision', 0),
                    'recall': performance_metrics.get('recall', 0),
                    'f1_score': performance_metrics.get('f1_score', 0)
                }
            else:  # Regression
                ui_metrics = {
                    'r2_score': performance_metrics.get('r2_score', 0),
                    'mse': performance_metrics.get('mse', 0),
                    'rmse': performance_metrics.get('rmse', 0),
                    'mae': performance_metrics.get('mae', 0)
                }
            
            training_details = {
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features': X.shape[1],
                'training_time': round(training_time, 2),
                'total_samples': len(X),
                'feature_names': original_feature_names,  # Use original names, not processed ones
                'target_column': target_column,
                'model_type': scenario,
                'cv_folds': cv_folds,
                'best_score': float(grid_search.best_score_)
            }
            
            return {
                'success': True,
                'model_name': original_name,
                'model_folder': model_folder,
                'training_time': training_time,
                'test_score': performance_metrics.get('accuracy' if scenario == "classification" else 'r2_score', 0.0),
                'cv_score': float(grid_search.best_score_),
                'score_name': 'accuracy' if scenario == "classification" else 'r2_score',
                'best_params': grid_search.best_params_,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'scenario': scenario,
                'performance_metrics': performance_metrics,
                'performance': ui_metrics,  # UI-friendly metrics
                'training_details': training_details,  # Training info for UI
                'feature_info': {
                    'feature_names': original_feature_names,  # Use original feature names
                    'target_column': target_column,
                    'problem_type': scenario,
                    'original_feature_names': original_feature_names,  # Preserve original names
                    'feature_count': X.shape[1],
                    'dataset_shape': X.shape,
                    'numeric_cols': numeric_features,      # Add actual numeric columns from data  
                    'categorical_cols': categorical_features  # Add actual categorical columns from data
                },  # Feature information for frontend
                'model_info': {
                    'model_directory': model_folder,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'feature_count': X.shape[1],
                    'timestamp': timestamp,
                    'feature_names': original_feature_names  # Also include here for completeness
                }  # Additional model information
            }
            
        except Exception as e:
            print(f"\n{'='*100}")
            print(f"Γ¥î TRAINING FAILED FOR {original_name.upper()}")
            print(f"{'='*100}")
            print(f"Error: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            print(f"{'='*100}\n")
            
            return {
                'success': False,
                'error': str(e),
                'model_name': original_name,
                'training_time': 0
            }
            print(f"≡ƒöì Parameter grid size: {len(param_grid)} parameters")
            
            # 7. Hyperparameter Tuning with GridSearchCV
            print(f"\n≡ƒöì STEP 6: HYPERPARAMETER OPTIMIZATION")
            print(f"{'='*50}")
            
            scoring = 'accuracy' if scenario == 'classification' else 'r2'
            
            # Use RandomizedSearchCV for faster training on large parameter spaces
            search_method = RandomizedSearchCV if len(param_grid) > 20 else GridSearchCV
            
            grid_search = search_method(
                pipeline,
                param_grid,
                cv=5,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                n_iter=15 if search_method == RandomizedSearchCV else None
            )
            
            print(f"≡ƒöº Using {search_method.__name__} with 5-fold cross-validation")
            print(f"≡ƒôè Scoring metric: {scoring}")
            
            # 8. Model Training with Timing
            print(f"\n≡ƒÜÇ STEP 7: MODEL TRAINING")
            print(f"{'='*50}")
            
            training_start = time.time()
            grid_search.fit(X_train, y_train)
            training_time = time.time() - training_start
            
            print(f"Γ£à Training completed in {training_time:.2f} seconds")
            print(f"≡ƒÅå Best CV score: {grid_search.best_score_:.4f}")
            print(f"≡ƒöº Best parameters:")
            for param, value in grid_search.best_params_.items():
                print(f"   {param}: {value}")
            
            # 9. Model Evaluation
            print(f"\n≡ƒôê STEP 8: MODEL EVALUATION")
            print(f"{'='*50}")
            
            y_pred = grid_search.predict(X_test)
            
            if scenario == 'classification':
                from sklearn.metrics import accuracy_score, classification_report
                test_score = accuracy_score(y_test, y_pred)
                print(f"≡ƒÄ» Test Accuracy: {test_score:.4f} ({test_score*100:.2f}%)")
                print(f"\n≡ƒôï Classification Report:")
                print(classification_report(y_test, y_pred))
            else:
                from sklearn.metrics import mean_squared_error, r2_score
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                test_score = r2
                print(f"≡ƒÄ» Test R┬▓ Score: {r2:.4f} ({r2*100:.2f}%)")
                print(f"≡ƒôè Mean Squared Error: {mse:.4f}")
                print(f"≡ƒôè Root MSE: {np.sqrt(mse):.4f}")
            
            # 10. Model Saving
            print(f"\n≡ƒÆ╛ STEP 9: MODEL SAVING")
            print(f"{'='*50}")
            
            # Create model directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_model_name = original_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            model_folder = f"models/{safe_model_name}"
            
            os.makedirs(model_folder, exist_ok=True)
            
            # Save the trained model
            model_path = os.path.join(model_folder, f"model_{timestamp}.joblib")
            joblib.dump(grid_search.best_estimator_, model_path)
            
            # Save preprocessing info
            preprocessing_info = {
                'numeric_features': numeric_features,
                'categorical_features': categorical_features,
                'datetime_features': datetime_features,
                'feature_names': list(X.columns),
                'target_column': target_column,
                'scenario': scenario
            }
            
            preprocessing_path = os.path.join(model_folder, f"preprocessing_{timestamp}.json")
            with open(preprocessing_path, 'w') as f:
                import json
                json.dump(preprocessing_info, f, indent=2)
            
            # Save metadata
            metadata = {
                'model_name': original_name,
                'model_type': scenario,
                'timestamp': timestamp,
                'training_time': training_time,
                'best_cv_score': float(grid_search.best_score_),
                'test_score': float(test_score),
                'best_params': grid_search.best_params_,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'feature_names': original_feature_names,  # Add feature names (filtered)
                'target_column': target_column,  # Add target column
                'selected_columns': selected_columns if selected_columns else original_feature_names,  # Store user selection
                'score_name': 'accuracy' if scenario == 'classification' else 'r2_score'
            }
            
            metadata_path = os.path.join(model_folder, f"metadata_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Γ£à Model saved: {model_path}")
            print(f"Γ£à Preprocessing info saved: {preprocessing_path}")
            print(f"Γ£à Metadata saved: {metadata_path}")
            
            print(f"\n{'='*80}")
            print(f"≡ƒÄë COMPREHENSIVE TRAINING COMPLETED SUCCESSFULLY!")
            print(f"{'='*80}")
            print(f"≡ƒôü Model folder: {model_folder}")
            print(f"≡ƒòÉ Total training time: {training_time:.2f} seconds")
            print(f"≡ƒÄ» Final score: {test_score:.4f}")
            
            return {
                'success': True,
                'model_name': original_name,
                'model_folder': model_folder,
                'training_time': training_time,
                'test_score': float(test_score),
                'cv_score': float(grid_search.best_score_),
                'score_name': 'accuracy' if scenario == 'classification' else 'r2_score',
                'best_params': grid_search.best_params_,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'scenario': scenario
            }
            
        except Exception as e:
            print(f"\nΓ¥î TRAINING FAILED: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'model_name': original_name,
                'training_time': 0
            }

    def _get_model_and_params(self, model_name: str, scenario: str):
        """
        Get comprehensive model instance and parameter grid for ALL supported models
        
        Supports:
        - Classification: Random Forest, XGBoost, LightGBM, CatBoost, SVM, Logistic Regression, 
                         Neural Network, KNN, Decision Tree, Gradient Boosting, Naive Bayes
        - Regression: Random Forest, XGBoost, LightGBM, CatBoost, SVR, Linear Regression,
                     Ridge, Lasso, ElasticNet, Neural Network, Gradient Boosting
        - Clustering: KMeans, DBSCAN, Hierarchical Clustering
        
        Args:
            model_name (str): Model name to configure
            scenario (str): 'classification', 'regression', or 'clustering'
            
        Returns:
            tuple: (model_instance, param_grid)
        """
        # Import all required models
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.svm import SVC, SVR
        from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.naive_bayes import GaussianNB
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        
        model_name_lower = model_name.lower()
        print(f"≡ƒöº Configuring model: {model_name} for {scenario}")
        
        # ==================================================
        # CLASSIFICATION MODELS
        # ==================================================
        if scenario == 'classification':
            
            # Random Forest Classifier
            if 'random' in model_name_lower and 'forest' in model_name_lower:
                model = RandomForestClassifier(random_state=42, n_jobs=-1)
                param_grid = {
                    'model__n_estimators': [100, 200, 300],
                    'model__max_depth': [10, 20, None],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4]
                }
                print(f"   Γ£à Random Forest Classifier configured")
                
            # XGBoost Classifier
            elif 'xgb' in model_name_lower or 'xgboost' in model_name_lower:
                try:
                    from xgboost import XGBClassifier
                    model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1, verbosity=0)
                    param_grid = {
                        'model__n_estimators': [100, 200, 300],
                        'model__max_depth': [3, 6, 10],
                        'model__learning_rate': [0.01, 0.1, 0.2],
                        'model__subsample': [0.8, 0.9, 1.0]
                    }
                    print(f"   Γ£à XGBoost Classifier configured (native)")
                except ImportError:
                    print(f"   ΓÜá∩╕Å  XGBoost not available - using GradientBoosting as fallback")
                    model = GradientBoostingClassifier(random_state=42)
                    param_grid = {
                        'model__n_estimators': [100, 200],
                        'model__max_depth': [3, 6],
                        'model__learning_rate': [0.01, 0.1]
                    }
                    
            # LightGBM Classifier  
            elif 'lightgbm' in model_name_lower or 'lgb' in model_name_lower:
                try:
                    from lightgbm import LGBMClassifier
                    model = LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1, force_row_wise=True)
                    param_grid = {
                        'model__n_estimators': [100, 200, 300],
                        'model__num_leaves': [31, 50, 100],
                        'model__learning_rate': [0.05, 0.1, 0.15],
                        'model__feature_fraction': [0.8, 0.9, 1.0]
                    }
                    print(f"   Γ£à LightGBM Classifier configured (native)")
                except ImportError:
                    print(f"   ΓÜá∩╕Å  LightGBM not available - using GradientBoosting as fallback")
                    model = GradientBoostingClassifier(random_state=42)
                    param_grid = {
                        'model__n_estimators': [100, 200],
                        'model__max_depth': [3, 6],
                        'model__learning_rate': [0.01, 0.1]
                    }
                    
            # CatBoost Classifier
            elif 'catboost' in model_name_lower:
                try:
                    from catboost import CatBoostClassifier
                    model = CatBoostClassifier(random_state=42, verbose=False, thread_count=-1)
                    param_grid = {
                        'model__iterations': [100, 200, 300],
                        'model__depth': [4, 6, 8],
                        'model__learning_rate': [0.05, 0.1, 0.15],
                        'model__l2_leaf_reg': [1, 3, 5]
                    }
                    print(f"   Γ£à CatBoost Classifier configured (native)")
                except ImportError:
                    print(f"   ΓÜá∩╕Å  CatBoost not available - using RandomForest as fallback")
                    model = RandomForestClassifier(random_state=42, n_jobs=-1)
                    param_grid = {
                        'model__n_estimators': [100, 200],
                        'model__max_depth': [10, 20],
                        'model__min_samples_split': [2, 5]
                    }
                    
            # Support Vector Machine
            elif 'svm' in model_name_lower or 'support vector' in model_name_lower:
                model = SVC(random_state=42, probability=True)
                param_grid = {
                    'model__C': [0.1, 1, 10, 100],
                    'model__kernel': ['linear', 'rbf'],
                    'model__gamma': ['scale', 'auto']
                }
                print(f"   Γ£à Support Vector Classifier configured")
                
            # Logistic Regression
            elif 'logistic' in model_name_lower:
                model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
                param_grid = {
                    'model__C': [0.01, 0.1, 1, 10, 100],
                    'model__penalty': ['l2'],
                    'model__solver': ['lbfgs', 'liblinear']
                }
                print(f"   Γ£à Logistic Regression configured")
                
            # Neural Network (MLP)
            elif 'neural' in model_name_lower or 'mlp' in model_name_lower:
                model = MLPClassifier(random_state=42, max_iter=1000)
                param_grid = {
                    'model__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'model__activation': ['relu', 'tanh'],
                    'model__alpha': [0.0001, 0.001, 0.01],
                    'model__learning_rate': ['constant', 'adaptive']
                }
                print(f"   Γ£à Neural Network Classifier configured")
                
            # K-Nearest Neighbors
            elif 'knn' in model_name_lower or 'neighbor' in model_name_lower or 'k-neighbor' in model_name_lower:
                model = KNeighborsClassifier(n_jobs=-1)
                param_grid = {
                    'model__n_neighbors': [3, 5, 7, 11, 15],
                    'model__weights': ['uniform', 'distance'],
                    'model__metric': ['euclidean', 'manhattan']
                }
                print(f"   Γ£à K-Nearest Neighbors Classifier configured")
                
            # Decision Tree
            elif 'decision' in model_name_lower and 'tree' in model_name_lower:
                model = DecisionTreeClassifier(random_state=42)
                param_grid = {
                    'model__max_depth': [5, 10, 20, None],
                    'model__min_samples_split': [2, 5, 10, 20],
                    'model__min_samples_leaf': [1, 2, 4, 8],
                    'model__criterion': ['gini', 'entropy']
                }
                print(f"   Γ£à Decision Tree Classifier configured")
                
            # Gradient Boosting
            elif 'gradient' in model_name_lower and 'boost' in model_name_lower:
                model = GradientBoostingClassifier(random_state=42)
                param_grid = {
                    'model__n_estimators': [100, 200, 300],
                    'model__max_depth': [3, 5, 7],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__subsample': [0.8, 0.9, 1.0]
                }
                print(f"   Γ£à Gradient Boosting Classifier configured")
                
            # Naive Bayes
            elif 'naive' in model_name_lower or 'bayes' in model_name_lower:
                model = GaussianNB()
                param_grid = {
                    'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                }
                print(f"   Γ£à Naive Bayes Classifier configured")
                
            # Default fallback
            else:
                print(f"   ΓÜá∩╕Å  Unknown classification model '{model_name}', using Random Forest as default")
                model = RandomForestClassifier(random_state=42, n_jobs=-1)
                param_grid = {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [10, 20],
                    'model__min_samples_split': [2, 5]
                }
        
        # ==================================================
        # REGRESSION MODELS
        # ==================================================
        elif scenario == 'regression':
            
            # Random Forest Regressor
            if 'random' in model_name_lower and 'forest' in model_name_lower:
                model = RandomForestRegressor(random_state=42, n_jobs=-1)
                param_grid = {
                    'model__n_estimators': [100, 200, 300],
                    'model__max_depth': [10, 20, None],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4]
                }
                print(f"   Γ£à Random Forest Regressor configured")
                
            # XGBoost Regressor
            elif 'xgb' in model_name_lower or 'xgboost' in model_name_lower:
                try:
                    from xgboost import XGBRegressor
                    model = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
                    param_grid = {
                        'model__n_estimators': [100, 200, 300],
                        'model__max_depth': [3, 6, 10],
                        'model__learning_rate': [0.01, 0.1, 0.2],
                        'model__subsample': [0.8, 0.9, 1.0]
                    }
                    print(f"   Γ£à XGBoost Regressor configured (native)")
                except ImportError:
                    print(f"   ΓÜá∩╕Å  XGBoost not available - using GradientBoosting as fallback")
                    model = GradientBoostingRegressor(random_state=42)
                    param_grid = {
                        'model__n_estimators': [100, 200],
                        'model__max_depth': [3, 6],
                        'model__learning_rate': [0.01, 0.1]
                    }
                    
            # LightGBM Regressor
            elif 'lightgbm' in model_name_lower or 'lgb' in model_name_lower:
                try:
                    from lightgbm import LGBMRegressor
                    model = LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1, force_row_wise=True)
                    param_grid = {
                        'model__n_estimators': [100, 200, 300],
                        'model__num_leaves': [31, 50, 100],
                        'model__learning_rate': [0.05, 0.1, 0.15],
                        'model__feature_fraction': [0.8, 0.9, 1.0]
                    }
                    print(f"   Γ£à LightGBM Regressor configured (native)")
                except ImportError:
                    print(f"   ΓÜá∩╕Å  LightGBM not available - using GradientBoosting as fallback")
                    model = GradientBoostingRegressor(random_state=42)
                    param_grid = {
                        'model__n_estimators': [100, 200],
                        'model__max_depth': [3, 6],
                        'model__learning_rate': [0.01, 0.1]
                    }
                    
            # CatBoost Regressor
            elif 'catboost' in model_name_lower:
                try:
                    from catboost import CatBoostRegressor
                    model = CatBoostRegressor(random_state=42, verbose=False, thread_count=-1)
                    param_grid = {
                        'model__iterations': [100, 200, 300],
                        'model__depth': [4, 6, 8],
                        'model__learning_rate': [0.05, 0.1, 0.15],
                        'model__l2_leaf_reg': [1, 3, 5]
                    }
                    print(f"   Γ£à CatBoost Regressor configured (native)")
                except ImportError:
                    print(f"   ΓÜá∩╕Å  CatBoost not available - using RandomForest as fallback")
                    model = RandomForestRegressor(random_state=42, n_jobs=-1)
                    param_grid = {
                        'model__n_estimators': [100, 200],
                        'model__max_depth': [10, 20],
                        'model__min_samples_split': [2, 5]
                    }
                    
            # Support Vector Regressor
            elif 'svm' in model_name_lower or 'support vector' in model_name_lower:
                model = SVR()
                param_grid = {
                    'model__C': [0.1, 1, 10, 100],
                    'model__kernel': ['linear', 'rbf'],
                    'model__gamma': ['scale', 'auto']
                }
                print(f"   Γ£à Support Vector Regressor configured")
                
            # Linear Regression
            elif 'linear' in model_name_lower and 'regression' in model_name_lower:
                model = LinearRegression(n_jobs=-1)
                param_grid = {
                    'model__fit_intercept': [True, False],
                    'model__positive': [True, False]
                }
                print(f"   Γ£à Linear Regression configured")
                
            # Ridge Regression
            elif 'ridge' in model_name_lower:
                model = Ridge(random_state=42)
                param_grid = {
                    'model__alpha': [0.1, 1.0, 10.0, 100.0],
                    'model__fit_intercept': [True, False],
                    'model__solver': ['auto', 'svd', 'cholesky', 'lsqr']
                }
                print(f"   Γ£à Ridge Regression configured")
                
            # Lasso Regression
            elif 'lasso' in model_name_lower:
                model = Lasso(random_state=42, max_iter=2000)
                param_grid = {
                    'model__alpha': [0.01, 0.1, 1.0, 10.0],
                    'model__fit_intercept': [True, False],
                    'model__selection': ['cyclic', 'random']
                }
                print(f"   Γ£à Lasso Regression configured")
                
            # ElasticNet
            elif 'elastic' in model_name_lower or 'elasticnet' in model_name_lower:
                model = ElasticNet(random_state=42, max_iter=2000)
                param_grid = {
                    'model__alpha': [0.01, 0.1, 1.0, 10.0],
                    'model__l1_ratio': [0.1, 0.5, 0.7, 0.9],
                    'model__fit_intercept': [True, False]
                }
                print(f"   Γ£à ElasticNet Regression configured")
                
            # Gradient Boosting Regressor
            elif 'gradient' in model_name_lower and 'boost' in model_name_lower:
                model = GradientBoostingRegressor(random_state=42)
                param_grid = {
                    'model__n_estimators': [100, 200, 300],
                    'model__max_depth': [3, 5, 7],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__subsample': [0.8, 0.9, 1.0]
                }
                print(f"   Γ£à Gradient Boosting Regressor configured")
                
            # Neural Network Regressor
            elif 'neural' in model_name_lower or 'mlp' in model_name_lower:
                model = MLPRegressor(random_state=42, max_iter=1000)
                param_grid = {
                    'model__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'model__activation': ['relu', 'tanh'],
                    'model__alpha': [0.0001, 0.001, 0.01],
                    'model__learning_rate': ['constant', 'adaptive']
                }
                print(f"   Γ£à Neural Network Regressor configured")
                
            # Default fallback
            else:
                print(f"   ΓÜá∩╕Å  Unknown regression model '{model_name}', using Random Forest as default")
                model = RandomForestRegressor(random_state=42, n_jobs=-1)
                param_grid = {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [10, 20],
                    'model__min_samples_split': [2, 5]
                }
        
        # ==================================================
        # CLUSTERING MODELS
        # ==================================================
        elif scenario == 'clustering':
            
            # KMeans
            if 'kmeans' in model_name_lower or 'k-means' in model_name_lower:
                model = KMeans(random_state=42, n_init=10)
                param_grid = {
                    'model__n_clusters': [2, 3, 4, 5, 6, 7, 8],
                    'model__init': ['k-means++', 'random'],
                    'model__algorithm': ['lloyd', 'elkan']
                }
                print(f"   Γ£à KMeans Clustering configured")
                
            # DBSCAN
            elif 'dbscan' in model_name_lower:
                model = DBSCAN()
                param_grid = {
                    'model__eps': [0.3, 0.5, 0.7, 1.0, 1.5],
                    'model__min_samples': [3, 5, 7, 10],
                    'model__metric': ['euclidean', 'manhattan']
                }
                print(f"   Γ£à DBSCAN Clustering configured")
                
            # Hierarchical/Agglomerative Clustering
            elif 'hierarchical' in model_name_lower or 'agglomerative' in model_name_lower:
                model = AgglomerativeClustering()
                param_grid = {
                    'model__n_clusters': [2, 3, 4, 5, 6, 7, 8],
                    'model__linkage': ['ward', 'complete', 'average', 'single'],
                    'model__metric': ['euclidean', 'manhattan']
                }
                print(f"   Γ£à Hierarchical Clustering configured")
                
            # Default fallback
            else:
                print(f"   ΓÜá∩╕Å  Unknown clustering model '{model_name}', using KMeans as default")
                model = KMeans(random_state=42, n_init=10)
                param_grid = {
                    'model__n_clusters': [3, 4, 5],
                    'model__init': ['k-means++']
                }
        
        # Default fallback for any unexpected scenario
        else:
            print(f"   ΓÜá∩╕Å  Unknown scenario '{scenario}', using Random Forest Classifier as default")
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {
                'model__n_estimators': [100, 200],
                'model__max_depth': [10, 20]
            }
        
        # Calculate parameter combinations
        total_combinations = 1
        for param_values in param_grid.values():
            total_combinations *= len(param_values) if isinstance(param_values, list) else 1
            
        print(f"   ≡ƒöó Parameter combinations to test: {total_combinations}")
        
        return model, param_grid

# Create a global instance
ml_core = MLCore()
