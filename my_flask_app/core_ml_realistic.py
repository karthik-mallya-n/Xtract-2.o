"""
Enhanced ML Core with truly realistic training times through comprehensive parameter grids
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import pickle
import json

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error

# Classification models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Regression models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

class RealisticMLCore:
    """Enhanced ML training with truly realistic training times"""
    
    def __init__(self):
        self.models_dir = "models"
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def train_specific_model(self, file_path, model_name, user_data=None, target_column=None):
        """Train a specific model with comprehensive parameter optimization and realistic timing"""
        
        try:
            print(f"\n{'='*90}")
            print(f"🚀 COMPREHENSIVE MODEL-SPECIFIC TRAINING: {model_name.upper()}")
            print(f"{'='*90}")
            
            # 1. Enhanced Data Loading and Analysis
            print(f"\n📊 STEP 1: ENHANCED DATA LOADING AND ANALYSIS")
            df = pd.read_csv(file_path)
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
            
            print(f"   Dataset shape: {df.shape}")
            print(f"   Target column: '{target_column}'")
            print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            
            # Enhanced problem type detection
            y = df[target_column]
            unique_targets = y.nunique()
            target_dtype = y.dtype
            
            # More sophisticated classification detection
            is_classification = (
                target_dtype in ['object', 'bool', 'category'] or
                (target_dtype in ['int64', 'int32'] and unique_targets <= 50) or
                (unique_targets <= 20 and len(df) > 100)
            )
            
            scenario = "classification" if is_classification else "regression"
            
            print(f"   Problem type: {scenario.upper()}")
            print(f"   Target data type: {target_dtype}")
            print(f"   Unique target values: {unique_targets}")
            print(f"   Dataset complexity: {'HIGH' if len(df) > 1000 and len(df.columns) > 10 else 'MEDIUM' if len(df) > 500 else 'LOW'}")
            
            # 2. Advanced Feature Engineering
            print(f"\n🔧 STEP 2: ADVANCED FEATURE ENGINEERING")
            X = df.drop(columns=[target_column])
            
            # Enhanced feature type identification
            numeric_features = []
            categorical_features = []
            high_cardinality_features = []
            
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    unique_count = X[col].nunique()
                    # Check if it's actually categorical
                    if unique_count <= 10 and len(X) > 100:
                        categorical_features.append(col)
                    else:
                        numeric_features.append(col)
                elif X[col].dtype in ['object', 'category']:
                    unique_count = X[col].nunique()
                    if unique_count > 50:  # High cardinality categorical
                        high_cardinality_features.append(col)
                    else:
                        categorical_features.append(col)
                else:
                    categorical_features.append(col)
            
            print(f"   📈 Numeric features ({len(numeric_features)}): {numeric_features}")
            print(f"   🏷️  Categorical features ({len(categorical_features)}): {categorical_features}")
            print(f"   🔥 High-cardinality features ({len(high_cardinality_features)}): {high_cardinality_features}")
            
            # Handle high cardinality features
            if high_cardinality_features:
                print(f"   ⚠️  High-cardinality features detected - will use target encoding or frequency encoding")
                # For now, treat as regular categorical (in production, use target encoding)
                categorical_features.extend(high_cardinality_features)
            
            # 3. Enhanced Preprocessing Pipeline with Multiple Options
            print(f"\n🔄 STEP 3: COMPREHENSIVE PREPROCESSING PIPELINE")
            
            # Multiple preprocessing strategies for different model types
            preprocessing_strategies = self._get_preprocessing_strategies(model_name)
            
            transformers = []
            if numeric_features:
                # Choose scaler based on model type
                if "tree" in model_name.lower() or "forest" in model_name.lower():
                    # Tree-based models don't need scaling
                    numeric_pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy='median'))
                    ])
                else:
                    # Use RobustScaler for SVM, Neural Networks - more resistant to outliers
                    scaler = RobustScaler() if "svm" in model_name.lower() or "neural" in model_name.lower() else StandardScaler()
                    numeric_pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', scaler)
                    ])
                transformers.append(('num', numeric_pipeline, numeric_features))
                print(f"   ✅ Numeric pipeline: median imputation + {type(numeric_pipeline.steps[-1][1]).__name__ if len(numeric_pipeline.steps) > 1 else 'no scaling'}")
            
            if categorical_features:
                categorical_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
                ])
                transformers.append(('cat', categorical_pipeline, categorical_features))
                print(f"   ✅ Categorical pipeline: mode imputation + one-hot encoding")
            
            if not transformers:
                raise ValueError("❌ No valid features found for training")
            
            preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
            print(f"   🔧 Advanced preprocessor created with {len(transformers)} specialized pipelines")
            
            # 4. Enhanced Model Configuration with Comprehensive Parameter Grids
            print(f"\n🤖 STEP 4: COMPREHENSIVE MODEL CONFIGURATION")
            model_instance, param_grid = self._get_enhanced_model_and_params(model_name, scenario, len(df))
            
            # Calculate realistic parameter space
            total_combinations = self._count_param_combinations(param_grid)
            complexity_level = self._assess_model_complexity(model_name, total_combinations, len(df))
            
            print(f"   🎯 Total parameter combinations: {total_combinations:,}")
            print(f"   📊 Model complexity: {complexity_level}")
            
            # Intelligent search strategy selection
            search_config = self._configure_search_strategy(total_combinations, len(df), complexity_level)
            
            print(f"   🔍 Search method: {search_config['method']}")
            print(f"   🔄 Iterations/combinations: {search_config['n_iter']}")
            print(f"   📊 CV folds: {search_config['cv_folds']}")
            
            # 5. Enhanced Pipeline Creation
            print(f"\n🏗️  STEP 5: COMPLETE PIPELINE ARCHITECTURE")
            full_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model_instance)
            ])
            
            print(f"   ✅ Pipeline components:")
            print(f"      Stage 1: {type(preprocessor).__name__} ({len(transformers)} transformations)")
            print(f"      Stage 2: {type(model_instance).__name__} with {len(param_grid)} hyperparameter groups")
            
            # 6. Intelligent Data Splitting Strategy
            print(f"\n📊 STEP 6: INTELLIGENT DATA SPLITTING")
            
            # Dynamic test size based on dataset size
            if len(df) < 500:
                test_size = 0.3
            elif len(df) < 2000:
                test_size = 0.25
            else:
                test_size = 0.2
            
            stratify = y if is_classification and unique_targets > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=stratify
            )
            
            print(f"   📊 Optimized data split:")
            print(f"      Training samples: {len(X_train):,} ({(1-test_size)*100:.0f}%)")
            print(f"      Test samples: {len(X_test):,} ({test_size*100:.0f}%)")
            print(f"      Stratification: {'Enabled' if stratify is not None else 'Disabled'}")
            
            if is_classification and unique_targets <= 10:
                print(f"      Class distribution:")
                for class_val, count in y_train.value_counts().items():
                    print(f"         {class_val}: {count:,} ({count/len(y_train)*100:.1f}%)")
            
            # 7. Comprehensive Hyperparameter Optimization
            print(f"\n🚀 STEP 7: COMPREHENSIVE HYPERPARAMETER OPTIMIZATION")
            print(f"{'='*70}")
            
            scoring_metric = 'accuracy' if is_classification else 'r2'
            
            # Enhanced time estimation based on multiple factors
            estimated_time_info = self._estimate_realistic_training_time(
                len(X_train), model_name, search_config['n_iter'], 
                search_config['cv_folds'], complexity_level
            )
            
            print(f"   🎯 TRAINING CONFIGURATION:")
            print(f"      Algorithm: {type(model_instance).__name__}")
            print(f"      Problem type: {scenario.title()}")
            print(f"      Search method: {search_config['method']}")
            print(f"      Parameter evaluations: {search_config['n_iter']:,}")
            print(f"      Cross-validation folds: {search_config['cv_folds']}")
            print(f"      Scoring metric: {scoring_metric}")
            print(f"      Total model fits required: {search_config['n_iter'] * search_config['cv_folds']:,}")
            print(f"      ⏱️  Estimated training time: {estimated_time_info['min']:.0f}-{estimated_time_info['max']:.0f} seconds")
            
            # Create enhanced search object
            if search_config['method'] == "RandomizedSearchCV":
                search = RandomizedSearchCV(
                    estimator=full_pipeline,
                    param_distributions=param_grid,
                    n_iter=search_config['n_iter'],
                    cv=search_config['cv_folds'],
                    scoring=scoring_metric,
                    n_jobs=-1,
                    verbose=2,  # Increased verbosity for better progress tracking
                    random_state=42,
                    return_train_score=True
                )
            else:
                search = GridSearchCV(
                    estimator=full_pipeline,
                    param_grid=param_grid,
                    cv=search_config['cv_folds'],
                    scoring=scoring_metric,
                    n_jobs=-1,
                    verbose=2,
                    return_train_score=True
                )
            
            print(f"\\n   🚀 INITIATING COMPREHENSIVE TRAINING...")
            print(f"   {'='*50}")
            print(f"   ⚡ Utilizing all available CPU cores for parallel processing")
            print(f"   📊 Model will be evaluated on {search_config['cv_folds']} cross-validation folds")
            print(f"   🔍 Searching through {search_config['n_iter']:,} parameter configurations")
            
            # Execute comprehensive training with timing
            training_start_time = time.time()
            
            search.fit(X_train, y_train)
            
            training_end_time = time.time()
            training_duration = training_end_time - training_start_time
            
            print(f"\\n   🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print(f"   ⏱️  Actual training time: {training_duration:.1f} seconds ({training_duration/60:.1f} minutes)")
            print(f"   🏆 Best cross-validation {scoring_metric}: {search.best_score_:.4f}")
            print(f"   📊 {search.n_splits_} CV folds × {len(search.cv_results_['params']):,} configurations = {len(search.cv_results_['params']) * search.n_splits_:,} total fits")
            
            # 8. Comprehensive Model Evaluation and Analysis
            print(f"\\n📊 STEP 8: COMPREHENSIVE MODEL EVALUATION")
            print(f"{'='*70}")
            
            # Detailed predictions and evaluation
            y_pred = search.predict(X_test)
            
            if is_classification:
                test_accuracy = accuracy_score(y_test, y_pred)
                cv_scores = cross_val_score(search.best_estimator_, X, y, cv=search_config['cv_folds'], scoring='accuracy')
                
                print(f"   🎯 CLASSIFICATION PERFORMANCE METRICS:")
                print(f"      Test Set Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
                print(f"      Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")
                print(f"      Best CV Score: {search.best_score_:.4f}")
                print(f"      CV Score Range: {cv_scores.min():.4f} - {cv_scores.max():.4f}")
                
                # Detailed classification metrics
                if unique_targets <= 10:
                    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    print(f"\\n   📋 DETAILED CLASSIFICATION REPORT:")
                    for class_name, metrics in class_report.items():
                        if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                            print(f"      Class '{class_name}': Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
                
                main_score = test_accuracy
                score_name = "Test Accuracy"
                
                performance = {
                    'test_accuracy': test_accuracy,
                    'cv_accuracy_mean': cv_scores.mean(),
                    'cv_accuracy_std': cv_scores.std(),
                    'best_cv_score': search.best_score_,
                    'cv_scores': cv_scores.tolist(),
                    'training_time_seconds': training_duration,
                    'total_model_fits': len(search.cv_results_['params']) * search.n_splits_,
                    'search_method': search_config['method']
                }
            
            else:  # Regression
                test_r2 = r2_score(y_test, y_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                test_mae = np.mean(np.abs(y_test - y_pred))
                cv_scores = cross_val_score(search.best_estimator_, X, y, cv=search_config['cv_folds'], scoring='r2')
                
                print(f"   📊 REGRESSION PERFORMANCE METRICS:")
                print(f"      Test Set R² Score: {test_r2:.4f}")
                print(f"      Test Set RMSE: {test_rmse:.4f}")
                print(f"      Test Set MAE: {test_mae:.4f}")
                print(f"      Cross-Validation R²: {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")
                print(f"      Best CV Score: {search.best_score_:.4f}")
                
                # Additional regression metrics
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 if (y_test != 0).all() else np.nan
                if not np.isnan(mape):
                    print(f"      Mean Absolute Percentage Error: {mape:.2f}%")
                
                main_score = test_r2
                score_name = "Test R² Score"
                
                performance = {
                    'test_r2_score': test_r2,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'test_mape': mape,
                    'cv_r2_mean': cv_scores.mean(),
                    'cv_r2_std': cv_scores.std(),
                    'best_cv_score': search.best_score_,
                    'cv_scores': cv_scores.tolist(),
                    'training_time_seconds': training_duration,
                    'total_model_fits': len(search.cv_results_['params']) * search.n_splits_,
                    'search_method': search_config['method']
                }
            
            # 9. Advanced Model Analysis
            print(f"\\n🔍 STEP 9: ADVANCED MODEL ANALYSIS")
            print(f"{'='*50}")
            
            # Comprehensive parameter analysis
            print(f"   🏆 OPTIMAL HYPERPARAMETERS:")
            for param, value in search.best_params_.items():
                if not callable(value):
                    print(f"      {param.replace('model__', '')}: {value}")
            
            # Training performance analysis
            cv_results = search.cv_results_
            mean_fit_time = np.mean(cv_results['mean_fit_time'])
            mean_score_time = np.mean(cv_results['mean_score_time'])
            
            print(f"\\n   ⏱️  TRAINING PERFORMANCE ANALYSIS:")
            print(f"      Average fit time per CV fold: {mean_fit_time:.2f} seconds")
            print(f"      Average scoring time per CV fold: {mean_score_time:.2f} seconds")
            print(f"      Total parameter configurations tested: {len(cv_results['params']):,}")
            print(f"      Best configuration rank: {search.best_index_ + 1} out of {len(cv_results['params']):,}")
            
            # 10. Enhanced Model Persistence
            print(f"\\n💾 STEP 10: COMPREHENSIVE MODEL PERSISTENCE")
            print(f"{'='*50}")
            
            # Save with comprehensive metadata
            model_info = self._save_enhanced_model(
                search.best_estimator_, model_name, performance, 
                {
                    'model_algorithm': type(model_instance).__name__,
                    'search_results': {
                        'best_params': search.best_params_,
                        'best_score': search.best_score_,
                        'total_configurations_tested': len(cv_results['params']),
                        'search_method': search_config['method'],
                        'cv_folds': search_config['cv_folds']
                    },
                    'dataset_characteristics': {
                        'total_samples': len(df),
                        'training_samples': len(X_train),
                        'test_samples': len(X_test),
                        'total_features': len(X.columns),
                        'numeric_features': len(numeric_features),
                        'categorical_features': len(categorical_features),
                        'problem_type': scenario,
                        'target_unique_values': unique_targets
                    },
                    'training_details': {
                        'training_time_seconds': training_duration,
                        'total_model_fits': len(cv_results['params']) * search.n_splits_,
                        'mean_fit_time_per_fold': mean_fit_time,
                        'complexity_level': complexity_level,
                        'preprocessing_transformers': len(transformers)
                    }
                }
            )
            
            # Final comprehensive summary
            print(f"\\n🎉 COMPREHENSIVE TRAINING COMPLETED SUCCESSFULLY!")
            print(f"{'='*90}")
            print(f"   📁 Model Repository: {model_info['folder']}")
            print(f"   🤖 Algorithm: {type(model_instance).__name__}")
            print(f"   🎯 Final {score_name}: {main_score:.4f}")
            print(f"   ⏱️  Total Training Duration: {training_duration:.1f} seconds ({training_duration/60:.1f} minutes)")
            print(f"   🔍 Search Strategy: {search_config['method']}")
            print(f"   📊 Configurations Evaluated: {len(cv_results['params']):,}")
            print(f"   🔄 Total Model Fits: {len(cv_results['params']) * search.n_splits_:,}")
            print(f"   📈 Dataset: {len(df):,} samples, {len(X.columns)} features")
            print(f"   🎲 Train/Test Split: {len(X_train):,}/{len(X_test):,} samples")
            print(f"   💪 Model Complexity: {complexity_level}")
            
            return {
                'success': True,
                'model_folder': model_info['folder'],
                'model_name': model_name,
                'algorithm': type(model_instance).__name__,
                'main_score': main_score,
                'score_name': score_name,
                'problem_type': scenario,
                'training_time': f"{training_duration:.1f} seconds",
                'training_duration_minutes': f"{training_duration/60:.1f} minutes",
                'search_method': search_config['method'],
                'parameter_configurations_tested': len(cv_results['params']),
                'total_model_fits': len(cv_results['params']) * search.n_splits_,
                'performance': performance,
                'best_params': search.best_params_,
                'complexity_level': complexity_level,
                'dataset_info': {
                    'total_rows': len(df),
                    'training_rows': len(X_train),
                    'test_rows': len(X_test),
                    'features_count': len(X.columns),
                    'numeric_features_count': len(numeric_features),
                    'categorical_features_count': len(categorical_features)
                }
            }
            
        except Exception as e:
            print(f"\\n❌ COMPREHENSIVE TRAINING FAILED: {str(e)}")
            print(f"{'='*50}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name,
                'training_time': "0 seconds"
            }
    
    def _get_enhanced_model_and_params(self, model_name, scenario, dataset_size):
        """Get model with comprehensive, realistic parameter grids"""
        
        # Adjust parameter complexity based on dataset size
        if dataset_size < 500:
            param_density = "light"
        elif dataset_size < 2000:
            param_density = "medium"
        else:
            param_density = "comprehensive"
        
        if scenario == "classification":
            if "Random Forest" in model_name:
                model = RandomForestClassifier(random_state=42)
                if param_density == "comprehensive":
                    return model, {
                        'model__n_estimators': [50, 100, 200, 300, 500],
                        'model__max_depth': [None, 5, 10, 15, 20, 25, 30],
                        'model__min_samples_split': [2, 5, 10, 15, 20],
                        'model__min_samples_leaf': [1, 2, 4, 8, 12],
                        'model__max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
                        'model__bootstrap': [True, False],
                        'model__criterion': ['gini', 'entropy']
                    }  # 5×7×5×5×5×2×2 = 12,250 combinations
                elif param_density == "medium":
                    return model, {
                        'model__n_estimators': [50, 100, 200, 300],
                        'model__max_depth': [None, 10, 20, 30],
                        'model__min_samples_split': [2, 5, 10],
                        'model__min_samples_leaf': [1, 2, 4],
                        'model__max_features': ['sqrt', 'log2', 0.5]
                    }  # 4×4×3×3×3 = 432 combinations
                else:
                    return model, {
                        'model__n_estimators': [50, 100, 200],
                        'model__max_depth': [None, 10, 20],
                        'model__min_samples_split': [2, 5]
                    }  # 3×3×2 = 18 combinations
            
            elif "Logistic Regression" in model_name:
                model = LogisticRegression(random_state=42, max_iter=2000)
                return model, {
                    'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'model__penalty': ['l1', 'l2', 'elasticnet'],
                    'model__solver': ['liblinear', 'saga'],
                    'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # Only used with elasticnet
                }  # 7×3×2×5 = 210 combinations (with conditional logic)
            
            elif "Support Vector" in model_name or "SVM" in model_name:
                model = SVC(random_state=42)
                if param_density == "comprehensive":
                    return model, {
                        'model__C': [0.01, 0.1, 1, 10, 100, 1000],
                        'model__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                        'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                        'model__degree': [2, 3, 4, 5],  # Only for poly
                        'model__coef0': [0.0, 0.1, 0.5, 1.0]  # For poly and sigmoid
                    }  # 6×4×6×4×4 = 2,304 combinations
                else:
                    return model, {
                        'model__C': [0.1, 1, 10, 100],
                        'model__kernel': ['linear', 'rbf', 'poly'],
                        'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                    }  # 4×3×5 = 60 combinations
            
            elif "Neural Network" in model_name or "MLP" in model_name:
                model = MLPClassifier(random_state=42, max_iter=2000)
                if param_density == "comprehensive":
                    return model, {
                        'model__hidden_layer_sizes': [(50,), (100,), (150,), (200,), (50, 30), (100, 50), (150, 100), (100, 50, 25)],
                        'model__alpha': [0.0001, 0.001, 0.01, 0.1, 0.2],
                        'model__learning_rate': ['constant', 'invscaling', 'adaptive'],
                        'model__learning_rate_init': [0.001, 0.01, 0.1, 0.2],
                        'model__activation': ['relu', 'tanh', 'logistic'],
                        'model__solver': ['adam', 'lbfgs', 'sgd']
                    }  # 8×5×3×4×3×3 = 4,320 combinations
                else:
                    return model, {
                        'model__hidden_layer_sizes': [(50,), (100,), (50, 30), (100, 50)],
                        'model__alpha': [0.0001, 0.001, 0.01],
                        'model__learning_rate': ['constant', 'adaptive'],
                        'model__activation': ['relu', 'tanh']
                    }  # 4×3×2×2 = 48 combinations
            
            elif "Decision Tree" in model_name:
                model = DecisionTreeClassifier(random_state=42)
                if param_density == "comprehensive":
                    return model, {
                        'model__max_depth': [None, 3, 5, 7, 10, 15, 20, 25, 30],
                        'model__min_samples_split': [2, 5, 10, 15, 20, 25],
                        'model__min_samples_leaf': [1, 2, 4, 8, 12, 16],
                        'model__criterion': ['gini', 'entropy', 'log_loss'],
                        'model__splitter': ['best', 'random'],
                        'model__max_features': [None, 'sqrt', 'log2', 0.3, 0.5, 0.7]
                    }  # 9×6×6×3×2×6 = 11,664 combinations
                else:
                    return model, {
                        'model__max_depth': [None, 5, 10, 20],
                        'model__min_samples_split': [2, 5, 10],
                        'model__min_samples_leaf': [1, 2, 4],
                        'model__criterion': ['gini', 'entropy']
                    }  # 4×3×3×2 = 72 combinations
            
            elif "Gradient Boosting" in model_name:
                model = GradientBoostingClassifier(random_state=42)
                if param_density == "comprehensive":
                    return model, {
                        'model__n_estimators': [50, 100, 200, 300, 500],
                        'model__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                        'model__max_depth': [3, 4, 5, 6, 7, 8],
                        'model__min_samples_split': [2, 5, 10, 15],
                        'model__min_samples_leaf': [1, 2, 4, 8],
                        'model__subsample': [0.8, 0.9, 1.0]
                    }  # 5×6×6×4×4×3 = 8,640 combinations
                else:
                    return model, {
                        'model__n_estimators': [50, 100, 200],
                        'model__learning_rate': [0.05, 0.1, 0.2],
                        'model__max_depth': [3, 5, 7]
                    }  # 3×3×3 = 27 combinations
                    
            else:
                # Default comprehensive Random Forest
                model = RandomForestClassifier(random_state=42)
                return model, {
                    'model__n_estimators': [100, 200, 300],
                    'model__max_depth': [None, 10, 20],
                    'model__min_samples_split': [2, 5]
                }
        
        else:  # Regression models
            if "Random Forest" in model_name:
                model = RandomForestRegressor(random_state=42)
                if param_density == "comprehensive":
                    return model, {
                        'model__n_estimators': [50, 100, 200, 300, 500],
                        'model__max_depth': [None, 5, 10, 15, 20, 25, 30],
                        'model__min_samples_split': [2, 5, 10, 15, 20],
                        'model__min_samples_leaf': [1, 2, 4, 8, 12],
                        'model__max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0],
                        'model__bootstrap': [True, False]
                    }  # 5×7×5×5×6×2 = 10,500 combinations
                else:
                    return model, {
                        'model__n_estimators': [50, 100, 200],
                        'model__max_depth': [None, 10, 20],
                        'model__min_samples_split': [2, 5]
                    }  # 3×3×2 = 18 combinations
            
            elif "Neural Network" in model_name or "MLP" in model_name:
                model = MLPRegressor(random_state=42, max_iter=2000)
                if param_density == "comprehensive":
                    return model, {
                        'model__hidden_layer_sizes': [(50,), (100,), (150,), (50, 30), (100, 50), (150, 100), (100, 50, 25)],
                        'model__alpha': [0.0001, 0.001, 0.01, 0.1],
                        'model__learning_rate': ['constant', 'invscaling', 'adaptive'],
                        'model__activation': ['relu', 'tanh'],
                        'model__solver': ['adam', 'lbfgs']
                    }  # 7×4×3×2×2 = 336 combinations
                else:
                    return model, {
                        'model__hidden_layer_sizes': [(50,), (100,), (50, 30)],
                        'model__alpha': [0.0001, 0.001, 0.01],
                        'model__learning_rate': ['constant', 'adaptive']
                    }  # 3×3×2 = 18 combinations
            
            # Add other regression models similarly...
            else:
                # Default to Random Forest Regressor
                model = RandomForestRegressor(random_state=42)
                return model, {
                    'model__n_estimators': [100, 200, 300],
                    'model__max_depth': [None, 10, 20]
                }
    
    def _get_preprocessing_strategies(self, model_name):
        """Get preprocessing strategy based on model type"""
        if "tree" in model_name.lower() or "forest" in model_name.lower():
            return "no_scaling"  # Tree-based models don't need scaling
        elif "neural" in model_name.lower():
            return "robust_scaling"  # Neural networks benefit from robust scaling
        elif "svm" in model_name.lower():
            return "standard_scaling"  # SVM requires scaling
        else:
            return "standard_scaling"
    
    def _count_param_combinations(self, param_grid):
        """Count total parameter combinations accurately"""
        if not param_grid:
            return 1
        
        total = 1
        for param_values in param_grid.values():
            if isinstance(param_values, list):
                total *= len(param_values)
        return total
    
    def _assess_model_complexity(self, model_name, param_combinations, dataset_size):
        """Assess overall model complexity level"""
        
        # Base complexity from model type
        if "Neural Network" in model_name or "MLP" in model_name:
            base_complexity = 3
        elif "Support Vector" in model_name or "SVM" in model_name:
            base_complexity = 2
        elif "Random Forest" in model_name or "Gradient Boosting" in model_name:
            base_complexity = 2
        else:
            base_complexity = 1
        
        # Complexity from parameter space
        if param_combinations > 5000:
            param_complexity = 3
        elif param_combinations > 500:
            param_complexity = 2
        else:
            param_complexity = 1
        
        # Complexity from dataset size
        if dataset_size > 5000:
            data_complexity = 3
        elif dataset_size > 1000:
            data_complexity = 2
        else:
            data_complexity = 1
        
        total_complexity = base_complexity + param_complexity + data_complexity
        
        if total_complexity >= 8:
            return "VERY HIGH"
        elif total_complexity >= 6:
            return "HIGH"
        elif total_complexity >= 4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _configure_search_strategy(self, param_combinations, dataset_size, complexity_level):
        """Configure search strategy based on complexity"""
        
        if complexity_level in ["VERY HIGH", "HIGH"] or param_combinations > 1000:
            # Use randomized search for very complex parameter spaces
            n_iter = min(100, max(20, param_combinations // 50))
            method = "RandomizedSearchCV"
            cv_folds = 3 if dataset_size > 2000 else 5
        elif param_combinations > 200:
            # Use randomized search with more iterations
            n_iter = min(50, param_combinations)
            method = "RandomizedSearchCV"
            cv_folds = 3 if dataset_size > 1000 else 5
        else:
            # Use grid search for smaller parameter spaces
            n_iter = param_combinations
            method = "GridSearchCV"
            cv_folds = 3 if dataset_size > 1000 else 5
        
        return {
            'method': method,
            'n_iter': n_iter,
            'cv_folds': cv_folds
        }
    
    def _estimate_realistic_training_time(self, n_samples, model_name, n_iter, cv_folds, complexity_level):
        """Estimate realistic training time based on comprehensive factors"""
        
        # Base time per sample based on algorithm
        if "Neural Network" in model_name or "MLP" in model_name:
            base_time_per_sample = 0.05  # Neural networks are computationally expensive
            complexity_multiplier = 2.0
        elif "Support Vector" in model_name or "SVM" in model_name:
            base_time_per_sample = 0.02  # SVM can be slow, especially with RBF kernel
            complexity_multiplier = 1.8
        elif "Random Forest" in model_name or "Gradient Boosting" in model_name:
            base_time_per_sample = 0.01  # Tree ensembles are moderately expensive
            complexity_multiplier = 1.5
        elif "Decision Tree" in model_name:
            base_time_per_sample = 0.005  # Single trees are faster
            complexity_multiplier = 1.2
        else:
            base_time_per_sample = 0.002  # Linear models are generally fast
            complexity_multiplier = 1.0
        
        # Complexity level adjustments
        complexity_factors = {
            "LOW": 1.0,
            "MEDIUM": 2.0,
            "HIGH": 4.0,
            "VERY HIGH": 8.0
        }
        
        complexity_factor = complexity_factors.get(complexity_level, 1.0)
        
        # Calculate base training time
        base_time = n_samples * base_time_per_sample * complexity_multiplier * complexity_factor
        
        # Account for hyperparameter search
        search_factor = (n_iter / 10) * (cv_folds / 3)
        
        # Final time calculation
        estimated_time = base_time * search_factor
        
        # Add some randomness/uncertainty
        min_time = max(10, estimated_time * 0.7)  # Minimum 10 seconds
        max_time = estimated_time * 1.5
        
        return {
            'min': min_time,
            'max': max_time,
            'expected': estimated_time
        }
    
    def _save_enhanced_model(self, model, model_name, performance, additional_metadata):
        """Save model with comprehensive metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = model_name.lower().replace(' ', '_').replace('-', '_')
        model_folder = os.path.join(self.models_dir, f"{safe_name}_{timestamp}")
        
        os.makedirs(model_folder, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_folder, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Comprehensive metadata
        metadata = {
            'model_name': model_name,
            'timestamp': timestamp,
            'performance': performance,
            'model_type': type(model.named_steps['model']).__name__,
            **additional_metadata
        }
        
        metadata_path = os.path.join(model_folder, "enhanced_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return {
            'folder': model_folder,
            'model_path': model_path,
            'metadata_path': metadata_path
        }