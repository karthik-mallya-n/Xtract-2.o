"""
Flask Application for ML Platform Backend
Main API server handling file uploads, model recommendations, training, and predictions.
"""

import os
import uuid
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import joblib

# Import our custom modules
from core_ml import ml_core
from tasks import celery_app, train_model_task

# Load environment variables
load_dotenv()

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
    app.config['MODEL_STORAGE_PATH'] = os.getenv('MODEL_STORAGE_PATH', 'models')
    
    # Ensure upload and model directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_STORAGE_PATH'], exist_ok=True)
    
    # Configure CORS
    cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
    CORS(app, origins=cors_origins, supports_credentials=True)
    
    return app

app = create_app()

# File upload validation
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json'}

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global storage for file metadata (in production, use a database)
uploaded_files = {}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and user questionnaire data
    
    Expected form data:
    - file: The dataset file
    - is_labeled: 'labeled' or 'unlabeled'
    - data_type: 'continuous' or 'categorical'
    
    Returns:
    - JSON response with file_id and success status
    """
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'File type not allowed. Supported types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Get user questionnaire data
        is_labeled = request.form.get('is_labeled', '')
        data_type = request.form.get('data_type', '')
        
        if not is_labeled or not data_type:
            return jsonify({
                'success': False,
                'error': 'Missing questionnaire data. Please specify if data is labeled and data type.'
            }), 400
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Secure the filename and save file
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        filename = f"{file_id}.{file_extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(file_path)
        
        # Store file metadata
        uploaded_files[file_id] = {
            'file_id': file_id,
            'original_filename': original_filename,
            'filename': filename,
            'file_path': file_path,
            'upload_date': datetime.now().isoformat(),
            'file_size': os.path.getsize(file_path),
            'user_answers': {
                'is_labeled': is_labeled,
                'data_type': data_type
            }
        }
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'filename': original_filename,
            'file_size': uploaded_files[file_id]['file_size'],
            'message': 'File uploaded successfully'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Upload failed: {str(e)}'
        }), 500

@app.route('/api/recommend-model', methods=['GET'])
def recommend_model():
    """
    Get model recommendations based on uploaded dataset
    
    Query parameters:
    - file_id: ID of uploaded file
    
    Returns:
    - JSON response with model recommendations
    """
    try:
        file_id = request.args.get('file_id')
        
        if not file_id:
            return jsonify({
                'success': False,
                'error': 'file_id parameter is required'
            }), 400
        
        # Check if file exists
        if file_id not in uploaded_files:
            return jsonify({
                'success': False,
                'error': 'File not found. Please upload a file first.'
            }), 404
        
        file_info = uploaded_files[file_id]
        file_path = file_info['file_path']
        user_answers = file_info['user_answers']
        
        # Analyze the dataset
        dataset_analysis = ml_core.analyze_dataset(file_path)
        
        # Make LLM request for recommendations
        llm_response = ml_core.make_llm_request(user_answers, dataset_analysis)
        
        if not llm_response['success']:
            return jsonify({
                'success': False,
                'error': llm_response['error']
            }), 500
        
        # Prepare response
        response_data = {
            'success': True,
            'file_id': file_id,
            'dataset_info': {
                'total_rows': dataset_analysis['total_rows'],
                'total_columns': dataset_analysis['total_columns'],
                'numeric_columns': len(dataset_analysis['numeric_columns']),
                'categorical_columns': len(dataset_analysis['categorical_columns'])
            },
            'user_answers': user_answers,
            'recommendations': llm_response.get('recommendations', {}),
            'raw_llm_response': llm_response.get('raw_response', '')
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get recommendations: {str(e)}'
        }), 500

@app.route('/api/train', methods=['POST'])
def start_training():
    """
    Start model training as a background task
    
    Expected JSON body:
    - file_id: ID of uploaded file
    - model_name: Name of selected model
    
    Returns:
    - JSON response with task_id for tracking training progress
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'JSON body is required'
            }), 400
        
        file_id = data.get('file_id')
        model_name = data.get('model_name')
        
        if not file_id or not model_name:
            return jsonify({
                'success': False,
                'error': 'file_id and model_name are required'
            }), 400
        
        # Check if file exists
        if file_id not in uploaded_files:
            return jsonify({
                'success': False,
                'error': 'File not found. Please upload a file first.'
            }), 404
        
        file_info = uploaded_files[file_id]
        file_path = file_info['file_path']
        user_answers = file_info['user_answers']
        
        # Start training task
        task = train_model_task.delay(file_id, file_path, model_name, user_answers)
        
        # Store task info
        file_info['training_task_id'] = task.id
        file_info['training_started'] = datetime.now().isoformat()
        file_info['selected_model'] = model_name
        
        return jsonify({
            'success': True,
            'task_id': task.id,
            'file_id': file_id,
            'model_name': model_name,
            'message': 'Training started successfully'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to start training: {str(e)}'
        }), 500

@app.route('/api/training-status/<task_id>', methods=['GET'])
def get_training_status(task_id):
    """
    Get the status of a training task
    
    Parameters:
    - task_id: Celery task ID
    
    Returns:
    - JSON response with training progress and metrics
    """
    try:
        # Get task result
        task = celery_app.AsyncResult(task_id)
        
        response_data = {
            'task_id': task_id,
            'state': task.state
        }
        
        if task.state == 'PENDING':
            response_data.update({
                'progress': 0,
                'status': 'Task is waiting to start...',
                'is_complete': False
            })
        
        elif task.state == 'PROGRESS':
            response_data.update({
                'progress': task.info.get('progress', 0),
                'status': task.info.get('status', 'Training in progress...'),
                'current_step': task.info.get('current_step', ''),
                'accuracy': task.info.get('accuracy', 0),
                'precision': task.info.get('precision', 0),
                'recall': task.info.get('recall', 0),
                'f1_score': task.info.get('f1_score', 0),
                'is_complete': False
            })
        
        elif task.state == 'SUCCESS':
            result = task.result
            response_data.update({
                'progress': 100,
                'status': 'Training completed successfully!',
                'is_complete': True,
                'results': result
            })
        
        elif task.state == 'FAILURE':
            response_data.update({
                'progress': 0,
                'status': 'Training failed',
                'error': str(task.info),
                'is_complete': True
            })
        
        return jsonify({
            'success': True,
            **response_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get training status: {str(e)}'
        }), 500

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    """
    Make predictions using a trained model
    
    Expected JSON body:
    - file_id: ID of the dataset used for training
    - input_data: Dictionary with feature values
    
    Returns:
    - JSON response with prediction results
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'JSON body is required'
            }), 400
        
        file_id = data.get('file_id')
        input_data = data.get('input_data', {})
        
        if not file_id:
            return jsonify({
                'success': False,
                'error': 'file_id is required'
            }), 400
        
        if not input_data:
            return jsonify({
                'success': False,
                'error': 'input_data is required'
            }), 400
        
        # Find the most recent model for this file_id
        import glob
        model_files = glob.glob(f"{app.config['MODEL_STORAGE_PATH']}/model_{file_id}_*.joblib")
        
        if not model_files:
            return jsonify({
                'success': False,
                'error': 'No trained model found for this dataset. Please train a model first.'
            }), 404
        
        # Use the most recent model
        model_files.sort(reverse=True)
        model_path = model_files[0]
        
        # Load the model and preprocessing info
        model_data = joblib.load(model_path)
        model = model_data['model']
        preprocessing_info = model_data['preprocessing_info']
        
        # Prepare input data for prediction
        import pandas as pd
        import numpy as np
        
        # Create DataFrame from input data
        input_df = pd.DataFrame([input_data])
        
        # Apply the same preprocessing as during training
        feature_names = preprocessing_info['feature_names']
        
        # Ensure all required features are present (fill missing with 0)
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Select and order features correctly
        input_df = input_df[feature_names]
        
        # Apply label encoders for categorical features
        label_encoders = preprocessing_info.get('label_encoders', {})
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = encoder.transform(input_df[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    input_df[col] = 0
        
        # Apply scaling
        scaler = preprocessing_info.get('scaler')
        if scaler:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get prediction probabilities if available
        prediction_proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_scaled)[0]
            prediction_proba = {
                'probabilities': proba.tolist(),
                'confidence': float(np.max(proba))
            }
        
        # Decode prediction if it was encoded
        target_encoder = preprocessing_info.get('target_encoder')
        if target_encoder:
            try:
                prediction_decoded = target_encoder.inverse_transform([prediction])[0]
            except:
                prediction_decoded = prediction
        else:
            prediction_decoded = prediction
        
        return jsonify({
            'success': True,
            'prediction': str(prediction_decoded),
            'raw_prediction': float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
            'confidence': prediction_proba['confidence'] if prediction_proba else None,
            'probabilities': prediction_proba if prediction_proba else None,
            'model_info': {
                'model_name': model_data.get('model_name', 'Unknown'),
                'training_date': model_data.get('training_date', ''),
                'accuracy': model_data.get('metrics', {}).get('accuracy', 0)
            },
            'input_data': input_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/models', methods=['GET'])
def list_models():
    """List all available trained models"""
    try:
        import glob
        model_files = glob.glob(f"{app.config['MODEL_STORAGE_PATH']}/*.joblib")
        
        models = []
        for model_path in model_files:
            try:
                model_data = joblib.load(model_path)
                models.append({
                    'filename': os.path.basename(model_path),
                    'model_name': model_data.get('model_name', 'Unknown'),
                    'training_date': model_data.get('training_date', ''),
                    'accuracy': model_data.get('metrics', {}).get('accuracy', 0),
                    'dataset_info': model_data.get('dataset_info', {})
                })
            except:
                continue
        
        return jsonify({
            'success': True,
            'models': models,
            'count': len(models)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to list models: {str(e)}'
        }), 500

@app.errorhandler(413)
def file_too_large(error):
    """Handle file size limit exceeded"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum file size is 16MB.'
    }), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Development server
    debug_mode = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    port = int(os.getenv('FLASK_PORT', 5000))
    
    print(f"Starting Flask server on port {port}")
    print(f"Debug mode: {debug_mode}")
    print(f"CORS origins: {os.getenv('CORS_ORIGINS', 'http://localhost:3000')}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)