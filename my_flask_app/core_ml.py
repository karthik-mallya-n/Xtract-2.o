"""
Core Machine Learning Module
Contains the main ML logic for model recommendations and data processing.
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MLCore:
    """Core ML functionality for the platform"""
    
    def __init__(self):
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.openrouter_base_url = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
    
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
        Make API request to OpenRouter for model recommendations
        
        Args:
            user_answers (dict): User's questionnaire responses
            dataset_analysis (dict): Dataset analysis results
            
        Returns:
            dict: LLM response with model recommendations
        """
        try:
            if not self.openrouter_api_key:
                raise ValueError("OpenRouter API key not configured")
            
            # Create the prompt
            prompt = self.create_llm_prompt(user_answers, dataset_analysis)
            
            # API request payload
            payload = {
                "model": "anthropic/claude-3.5-sonnet",  # You can change this model
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.3,
                "top_p": 0.9
            }
            
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:3000",  # Your frontend URL
                "X-Title": "ML Platform"
            }
            
            # Make the API request
            response = requests.post(
                f"{self.openrouter_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            llm_content = response_data['choices'][0]['message']['content']
            
            # Try to parse as JSON, fallback to raw text if needed
            try:
                recommendations = json.loads(llm_content)
                return {
                    'success': True,
                    'recommendations': recommendations,
                    'raw_response': llm_content
                }
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw response
                return {
                    'success': True,
                    'raw_response': llm_content,
                    'note': 'Response was not in JSON format'
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"API request failed: {str(e)}"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Unexpected error: {str(e)}"
            }
    
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

# Create a global instance
ml_core = MLCore()