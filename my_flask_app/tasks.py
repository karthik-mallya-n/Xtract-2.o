"""
Celery Tasks for Background Processing
Handles asynchronous model training and other long-running operations.
"""

import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from celery import Celery
from datetime import datetime
from typing import Dict, Any, Tuple
from core_ml import ml_core

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Celery app
def make_celery():
    """Create and configure Celery application"""
    celery = Celery('ml_platform')
    
    # Configure Celery
    celery.conf.update(
        broker_url=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
        result_backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=30 * 60,  # 30 minutes
        task_soft_time_limit=25 * 60,  # 25 minutes
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=1000,
    )
    
    return celery

celery_app = make_celery()

@celery_app.task(bind=True)
def train_model_task(self, file_id: str, file_path: str, model_name: str, user_answers: Dict[str, Any]):
    """
    Background task for training machine learning models
    
    Args:
        self: Celery task instance
        file_id (str): Unique identifier for the uploaded file
        file_path (str): Path to the dataset file
        model_name (str): Name of the model to train
        user_answers (dict): User's questionnaire responses
        
    Returns:
        dict: Training results including accuracy, precision, and model path
    """
    try:
        # Update task state to indicate training has started
        self.update_state(
            state='PROGRESS',
            meta={
                'current_step': 'Loading dataset',
                'progress': 10,
                'status': 'Loading and preprocessing data...'
            }
        )
        
        # Load the dataset
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current_step': 'Preprocessing data',
                'progress': 25,
                'status': 'Preparing data for training...'
            }
        )
        
        # Prepare data for training
        X, y, preprocessing_info = ml_core.prepare_data_for_training(df)
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current_step': 'Splitting data',
                'progress': 40,
                'status': 'Splitting dataset into train/test sets...'
            }
        )
        
        # Split data into training and testing sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current_step': 'Initializing model',
                'progress': 55,
                'status': f'Initializing {model_name} model...'
            }
        )
        
        # Get the model class and initialize
        ModelClass = ml_core.get_model_class(model_name)
        
        # Configure model parameters based on data characteristics
        model_params = get_model_parameters(model_name, X_train.shape, len(np.unique(y)))
        model = ModelClass(**model_params)
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current_step': 'Training model',
                'progress': 70,
                'status': 'Training model on your data...',
                'accuracy': 0,
                'precision': 0
            }
        )
        
        # Train the model with progress simulation
        start_time = time.time()
        
        # For models that support partial_fit or have n_estimators, we can show progress
        if hasattr(model, 'n_estimators'):
            # For ensemble methods, train incrementally to show progress
            original_n_estimators = model.n_estimators
            progress_steps = min(10, original_n_estimators)
            step_size = original_n_estimators // progress_steps
            
            for i in range(1, progress_steps + 1):
                model.n_estimators = i * step_size
                model.fit(X_train, y_train)
                
                # Calculate intermediate metrics
                y_pred = model.predict(X_test)
                accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
                
                progress = 70 + (i / progress_steps) * 20
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current_step': 'Training model',
                        'progress': int(progress),
                        'status': f'Training progress: {i}/{progress_steps} iterations...',
                        'accuracy': round(accuracy * 100, 2),
                        'precision': round(precision * 100, 2),
                        'recall': round(recall * 100, 2),
                        'f1_score': round(f1 * 100, 2),
                        'current_estimators': i * step_size
                    }
                )
                time.sleep(0.5)  # Simulate training time
            
            # Final training with full n_estimators
            model.n_estimators = original_n_estimators
            model.fit(X_train, y_train)
        else:
            # For other models, just train once
            model.fit(X_train, y_train)
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current_step': 'Evaluating model',
                'progress': 90,
                'status': 'Evaluating model performance...'
            }
        )
        
        # Make predictions and calculate final metrics
        y_pred = model.predict(X_test)
        accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
        
        # Calculate additional metrics
        training_time = time.time() - start_time
        
        # Save the trained model
        model_filename = f"model_{file_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        model_path = os.path.join('models', model_filename)
        
        # Save model and preprocessing info
        model_data = {
            'model': model,
            'preprocessing_info': preprocessing_info,
            'model_name': model_name,
            'training_date': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(df),
                'feature_count': X.shape[1],
                'target_classes': len(np.unique(y))
            },
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_time': training_time
            }
        }
        
        joblib.dump(model_data, model_path)
        
        # Return final results
        result = {
            'file_id': file_id,
            'model_name': model_name,
            'model_path': model_path,
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'training_time': round(training_time, 2),
            'dataset_info': model_data['dataset_info'],
            'training_date': model_data['training_date'],
            'status': 'completed',
            'is_complete': True
        }
        
        return result
        
    except Exception as e:
        # Handle errors
        error_msg = f"Training failed: {str(e)}"
        self.update_state(
            state='FAILURE',
            meta={
                'error': error_msg,
                'progress': 0,
                'status': 'Training failed'
            }
        )
        raise Exception(error_msg)

def get_model_parameters(model_name: str, data_shape: Tuple[int, int], n_classes: int) -> Dict[str, Any]:
    """
    Get optimized parameters for different models based on data characteristics
    
    Args:
        model_name (str): Name of the model
        data_shape (tuple): Shape of training data (n_samples, n_features)
        n_classes (int): Number of target classes
        
    Returns:
        dict: Model parameters
    """
    n_samples, n_features = data_shape
    
    # Base parameters for different models
    params = {}
    
    normalized_name = model_name.lower().replace(' ', '_').replace('-', '_')
    
    if 'random_forest' in normalized_name:
        params = {
            'n_estimators': min(100, max(10, n_samples // 100)),
            'max_depth': min(20, max(3, int(np.log2(n_features)) + 1)),
            'min_samples_split': max(2, n_samples // 1000),
            'min_samples_leaf': max(1, n_samples // 2000),
            'random_state': 42,
            'n_jobs': -1
        }
    
    elif 'svm' in normalized_name or 'svc' in normalized_name:
        params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'random_state': 42
        }
    
    elif 'logistic_regression' in normalized_name:
        params = {
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'liblinear' if n_features < 100 else 'lbfgs'
        }
    
    elif 'neural_network' in normalized_name or 'mlp' in normalized_name:
        hidden_layer_size = min(100, max(10, n_features * 2))
        params = {
            'hidden_layer_sizes': (hidden_layer_size,),
            'max_iter': 500,
            'random_state': 42,
            'early_stopping': True,
            'validation_fraction': 0.1
        }
    
    elif 'decision_tree' in normalized_name:
        params = {
            'max_depth': min(20, max(3, int(np.log2(n_features)) + 1)),
            'min_samples_split': max(2, n_samples // 1000),
            'min_samples_leaf': max(1, n_samples // 2000),
            'random_state': 42
        }
    
    return params

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Tuple of (accuracy, precision, recall, f1_score)
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Handle multi-class classification
    average = 'weighted' if len(np.unique(y_true)) > 2 else 'binary'
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    return accuracy, precision, recall, f1

@celery_app.task
def cleanup_old_files(days_old: int = 7):
    """
    Cleanup old uploaded files and models
    
    Args:
        days_old (int): Delete files older than this many days
    """
    import glob
    from datetime import datetime, timedelta
    
    cutoff_date = datetime.now() - timedelta(days=days_old)
    
    # Clean up upload files
    upload_files = glob.glob('uploads/*')
    for file_path in upload_files:
        file_stat = os.stat(file_path)
        file_date = datetime.fromtimestamp(file_stat.st_mtime)
        if file_date < cutoff_date:
            os.remove(file_path)
    
    # Clean up model files
    model_files = glob.glob('models/*')
    for file_path in model_files:
        file_stat = os.stat(file_path)
        file_date = datetime.fromtimestamp(file_stat.st_mtime)
        if file_date < cutoff_date:
            os.remove(file_path)

if __name__ == '__main__':
    # Run celery worker
    celery_app.start()