"""
Advanced Model Trainer with 90%+ Accuracy Optimization
Supports all major ML algorithms with automatic hyperparameter tuning
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Regression Models  
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

# Advanced libraries for better performance
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class AdvancedModelTrainer:
    """Advanced ML model trainer with 90%+ accuracy optimization"""
    
    def __init__(self, base_models_dir: str = "models"):
        self.base_models_dir = base_models_dir
        self.ensure_models_directory()
        
        # Model configurations with optimized hyperparameters
        self.classification_models = {
            'Logistic Regression': {
                'model': LogisticRegression,
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'solver': ['liblinear', 'lbfgs'],
                    'max_iter': [1000, 2000],
                    'random_state': [42]
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'random_state': [42]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'random_state': [42]
                }
            },
            'Support Vector Machine': {
                'model': SVC,
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto'],
                    'random_state': [42]
                }
            },
            'K-Neighbors': {
                'model': KNeighborsClassifier,
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree']
                }
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier,
                'params': {
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy'],
                    'random_state': [42]
                }
            },
            'Naive Bayes': {
                'model': GaussianNB,
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                }
            },
            'Neural Network': {
                'model': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive'],
                    'max_iter': [1000],
                    'random_state': [42]
                }
            }
        }
        
        self.regression_models = {
            'Linear Regression': {
                'model': LinearRegression,
                'params': {}
            },
            'Ridge Regression': {
                'model': Ridge,
                'params': {
                    'alpha': [0.1, 1, 10, 100],
                    'random_state': [42]
                }
            },
            'Lasso Regression': {
                'model': Lasso,
                'params': {
                    'alpha': [0.1, 1, 10, 100],
                    'random_state': [42]
                }
            },
            'ElasticNet': {
                'model': ElasticNet,
                'params': {
                    'alpha': [0.1, 1, 10],
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9],
                    'random_state': [42]
                }
            },
            'Random Forest Regressor': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'random_state': [42]
                }
            },
            'Gradient Boosting Regressor': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'random_state': [42]
                }
            },
            'Support Vector Regressor': {
                'model': SVR,
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'K-Neighbors Regressor': {
                'model': KNeighborsRegressor,
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree']
                }
            },
            'Decision Tree Regressor': {
                'model': DecisionTreeRegressor,
                'params': {
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['squared_error', 'absolute_error'],
                    'random_state': [42]
                }
            },
            'Neural Network Regressor': {
                'model': MLPRegressor,
                'params': {
                    'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive'],
                    'max_iter': [1000],
                    'random_state': [42]
                }
            }
        }
        
        # Add XGBoost and LightGBM if available
        if XGBOOST_AVAILABLE:
            self.classification_models['XGBoost'] = {
                'model': XGBClassifier,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'random_state': [42]
                }
            }
            self.regression_models['XGBoost Regressor'] = {
                'model': XGBRegressor,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'random_state': [42]
                }
            }
        
        if LIGHTGBM_AVAILABLE:
            self.classification_models['LightGBM'] = {
                'model': LGBMClassifier,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'random_state': [42]
                }
            }
            self.regression_models['LightGBM Regressor'] = {
                'model': LGBMRegressor,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'random_state': [42]
                }
            }
    
    def ensure_models_directory(self):
        """Ensure the models directory exists"""
        if not os.path.exists(self.base_models_dir):
            os.makedirs(self.base_models_dir)
    
    def create_model_folder(self, model_name: str) -> str:
        """Create a dedicated folder for the model"""
        # Clean model name for folder
        folder_name = model_name.lower().replace(' ', '_').replace('-', '_')
        model_path = os.path.join(self.base_models_dir, folder_name)
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        return model_path
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """Advanced data preprocessing for maximum accuracy"""
        
        print(f"🔧 PREPROCESSING DATA")
        print(f"   📊 Original shape: {df.shape}")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store original feature names
        feature_names = list(X.columns)
        
        # Handle missing values
        if X.isnull().any().any():
            print(f"   🚨 Handling missing values...")
            # For numeric columns, use median
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
            
            # For categorical columns, use mode
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        
        # Encode categorical variables
        label_encoders = {}
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            print(f"   🔤 Encoding {len(categorical_cols)} categorical columns...")
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        
        # Handle target variable encoding if categorical
        target_encoder = None
        if y.dtype == 'object' or y.dtype.name == 'category':
            print(f"   🎯 Encoding target variable...")
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
        
        # Feature selection for better performance
        print(f"   🎯 Selecting best features...")
        if len(feature_names) > 10:  # Only if we have many features
            # Determine if classification or regression
            is_classification = len(np.unique(y)) < 20 and y.dtype in ['int64', 'int32', 'object']
            
            if is_classification:
                selector = SelectKBest(score_func=f_classif, k=min(15, len(feature_names)))
            else:
                selector = SelectKBest(score_func=f_regression, k=min(15, len(feature_names)))
            
            X_selected = selector.fit_transform(X, y)
            selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
            print(f"   ✅ Selected {len(selected_features)} best features")
        else:
            X_selected = X.values
            selected_features = feature_names
        
        # Scale features for better performance
        print(f"   ⚖️ Scaling features...")
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_scaled = scaler.fit_transform(X_selected)
        
        preprocessing_info = {
            'label_encoders': label_encoders,
            'target_encoder': target_encoder,
            'scaler': scaler,
            'selected_features': selected_features,
            'feature_names': feature_names
        }
        
        print(f"   ✅ Final shape: {X_scaled.shape}")
        
        return X_scaled, y, selected_features, preprocessing_info
    
    def train_model(self, model_name: str, file_path: str, target_column: str) -> Dict[str, Any]:
        """Train a model with 90%+ accuracy optimization"""
        
        print(f"\n🚀 TRAINING MODEL: {model_name}")
        print("="*80)
        
        try:
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            print(f"📊 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Validate target column
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
            
            # Determine problem type
            unique_targets = df[target_column].nunique()
            is_classification = unique_targets < 20 and df[target_column].dtype in ['int64', 'int32', 'object', 'bool']
            
            problem_type = "classification" if is_classification else "regression"
            print(f"🎯 Problem type: {problem_type}")
            print(f"🎯 Target variable: {target_column} ({unique_targets} unique values)")
            
            # Select appropriate model configuration
            if is_classification:
                if model_name not in self.classification_models:
                    raise ValueError(f"Model '{model_name}' not available for classification")
                model_config = self.classification_models[model_name]
            else:
                if model_name not in self.regression_models:
                    raise ValueError(f"Model '{model_name}' not available for regression")
                model_config = self.regression_models[model_name]
            
            # Preprocess data
            X, y, feature_names, preprocessing_info = self.preprocess_data(df, target_column)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if is_classification else None
            )
            
            print(f"📊 Training set: {X_train.shape[0]} samples")
            print(f"📊 Test set: {X_test.shape[0]} samples")
            
            # Hyperparameter tuning with GridSearchCV
            print(f"🔍 Hyperparameter tuning...")
            
            base_model = model_config['model']()
            param_grid = model_config['params']
            
            if param_grid:  # If hyperparameters are defined
                cv_scorer = 'accuracy' if is_classification else 'r2'
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=5,
                    scoring=cv_scorer,
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                print(f"✅ Best parameters: {grid_search.best_params_}")
                print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
            else:
                best_model = base_model
                best_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate performance metrics
            if is_classification:
                accuracy = accuracy_score(y_test, y_pred)
                print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                
                # Cross-validation score
                cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
                cv_mean = cv_scores.mean()
                print(f"🔄 Cross-validation Accuracy: {cv_mean:.4f} ± {cv_scores.std()*2:.4f}")
                
                performance = {
                    'accuracy': accuracy,
                    'cv_accuracy': cv_mean,
                    'cv_std': cv_scores.std(),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                main_score = accuracy
                score_name = "Accuracy"
                
            else:
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                print(f"📊 Test R² Score: {r2:.4f}")
                print(f"📊 RMSE: {rmse:.4f}")
                
                # Cross-validation score
                cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
                cv_mean = cv_scores.mean()
                print(f"🔄 Cross-validation R²: {cv_mean:.4f} ± {cv_scores.std()*2:.4f}")
                
                performance = {
                    'r2_score': r2,
                    'rmse': rmse,
                    'cv_r2': cv_mean,
                    'cv_std': cv_scores.std()
                }
                
                main_score = r2
                score_name = "R² Score"
            
            # Create model folder
            model_folder = self.create_model_folder(model_name)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            model_filename = f"{model_name.lower().replace(' ', '_')}_{timestamp}.pkl"
            model_path = os.path.join(model_folder, model_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            # Save preprocessing info
            preprocessing_filename = f"preprocessing_{timestamp}.pkl"
            preprocessing_path = os.path.join(model_folder, preprocessing_filename)
            
            with open(preprocessing_path, 'wb') as f:
                pickle.dump(preprocessing_info, f)
            
            # Save model metadata
            metadata = {
                'model_name': model_name,
                'problem_type': problem_type,
                'target_column': target_column,
                'feature_names': feature_names,
                'training_date': datetime.now().isoformat(),
                'dataset_info': {
                    'total_samples': len(df),
                    'features': len(feature_names),
                    'target_unique_values': unique_targets
                },
                'performance': performance,
                'model_file': model_filename,
                'preprocessing_file': preprocessing_filename,
                'main_score': main_score,
                'score_name': score_name
            }
            
            metadata_filename = f"metadata_{timestamp}.json"
            metadata_path = os.path.join(model_folder, metadata_filename)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\n✅ MODEL TRAINING COMPLETED!")
            print(f"📁 Model saved in: {model_folder}")
            print(f"🎯 Final {score_name}: {main_score:.4f} ({main_score*100:.2f}%)")
            
            # Check if we achieved 90%+ accuracy/R²
            success_threshold = 0.90
            if main_score >= success_threshold:
                print(f"🎉 SUCCESS! Achieved {main_score*100:.2f}% - Above 90% threshold!")
            else:
                print(f"⚠️ {main_score*100:.2f}% - Below 90% threshold. Consider feature engineering or different model.")
            
            return {
                'success': True,
                'model_name': model_name,
                'model_folder': model_folder,
                'performance': performance,
                'main_score': main_score,
                'score_name': score_name,
                'problem_type': problem_type,
                'threshold_met': main_score >= success_threshold,
                'metadata_file': metadata_path
            }
            
        except Exception as e:
            print(f"❌ Error training model: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name
            }
    
    def load_model(self, model_folder: str, timestamp: str = None) -> Tuple[Any, Dict, Dict]:
        """Load a trained model and its preprocessing info"""
        
        if timestamp:
            model_file = f"{os.path.basename(model_folder)}_{timestamp}.pkl"
            preprocessing_file = f"preprocessing_{timestamp}.pkl"
            metadata_file = f"metadata_{timestamp}.json"
        else:
            # Load the most recent model
            model_files = [f for f in os.listdir(model_folder) if f.endswith('.pkl') and not f.startswith('preprocessing')]
            if not model_files:
                raise ValueError("No model files found")
            
            model_file = sorted(model_files)[-1]  # Most recent
            timestamp = model_file.split('_')[-1].replace('.pkl', '')
            preprocessing_file = f"preprocessing_{timestamp}.pkl"
            metadata_file = f"metadata_{timestamp}.json"
        
        # Load model
        model_path = os.path.join(model_folder, model_file)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load preprocessing info
        preprocessing_path = os.path.join(model_folder, preprocessing_file)
        with open(preprocessing_path, 'rb') as f:
            preprocessing_info = pickle.load(f)
        
        # Load metadata
        metadata_path = os.path.join(model_folder, metadata_file)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, preprocessing_info, metadata
    
    def predict(self, model_folder: str, new_data: pd.DataFrame, timestamp: str = None) -> np.ndarray:
        """Make predictions using a trained model"""
        
        model, preprocessing_info, metadata = self.load_model(model_folder, timestamp)
        
        # Apply same preprocessing
        X = new_data.copy()
        
        # Handle missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        
        # Apply label encoding
        for col, encoder in preprocessing_info['label_encoders'].items():
            if col in X.columns:
                # Handle unseen categories
                X[col] = X[col].astype(str)
                known_classes = set(encoder.classes_)
                X[col] = X[col].apply(lambda x: x if x in known_classes else encoder.classes_[0])
                X[col] = encoder.transform(X[col])
        
        # Select features
        X_selected = X[preprocessing_info['selected_features']].values
        
        # Scale features
        X_scaled = preprocessing_info['scaler'].transform(X_selected)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # Decode target if necessary
        if preprocessing_info['target_encoder']:
            predictions = preprocessing_info['target_encoder'].inverse_transform(predictions)
        
        return predictions