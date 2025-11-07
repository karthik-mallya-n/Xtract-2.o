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
# Task system disabled due to import issues
# from tasks import celery_app, train_model_task

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

    Resilience / Fallback Strategy:
    - If dataset analysis fails, returns minimal dataset_info and continues
    - If LLM call fails (missing key, network, parse error), returns backend static fallback models
    - Always returns HTTP 200 with success True unless request validation fails (400/404)
    - Warnings array included when fallbacks are used so the frontend can inform the user
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

        # Analyze the dataset with guard
        try:
            dataset_analysis = ml_core.analyze_dataset(file_path)
        except Exception as analysis_err:
            # Provide minimal dataset info fallback
            dataset_analysis = {
                'total_rows': 0,
                'total_columns': 0,
                'numeric_columns': [],
                'categorical_columns': []
            }
            analysis_warning = f"Dataset analysis failed: {analysis_err}"
        else:
            analysis_warning = None

        # Attempt LLM recommendations
        llm_response = ml_core.make_llm_request(user_answers, dataset_analysis)

        warnings = []
        if analysis_warning:
            warnings.append(analysis_warning)

        # Define backend fallback recommendations (must align with frontend expectations in select-model page)
        backend_fallback = {
            'recommended_models': [
                {
                    'name': 'Random Forest',
                    'description': 'Robust ensemble method suitable for mixed feature types and handles non-linear relationships well.',
                    'accuracy_estimate': 85,
                    'reasoning': 'Default choice when data characteristics are unknown or moderate sized.'
                }
            ],
            'alternative_models': [
                {
                    'name': 'Support Vector Machine',
                    'description': 'Effective in high dimensional spaces; good baseline for classification.',
                    'accuracy_estimate': 82
                },
                {
                    'name': 'Logistic Regression',
                    'description': 'Interpretable linear baseline useful for quick iteration.',
                    'accuracy_estimate': 78
                }
            ]
        }

        if not llm_response.get('success', False):
            error_msg = llm_response.get('error', 'LLM recommendation failed')
            warnings.append(f"AI recommendation failed: {error_msg}")
            print(f"⚠️ Using fallback recommendations due to AI failure: {error_msg}")
            recommendations = backend_fallback
            raw_llm_response = ''
            scenario_info = None
            semantic_analysis = None
        else:
            # Handle the new comprehensive JSON structure
            recs = llm_response.get('recommendations') or {}
            
            # Extract scenario and semantic analysis
            scenario_info = recs.get('scenario_detected', {})
            semantic_analysis = recs.get('semantic_analysis', {})
            
            # Process ranked models
            ranked_models = recs.get('recommended_models', [])
            primary_recommendation = recs.get('primary_recommendation', {})
            
            if ranked_models:
                # Convert ranked models to expected format
                recommended_models = []
                alternative_models = []
                
                for i, model in enumerate(ranked_models):
                    # Create a unique ID based on rank and name to avoid duplicates
                    model_name = model.get('name', f'Model {i+1}')
                    unique_id = f"model_{i+1}_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('-', '_')}"
                    
                    model_data = {
                        'id': unique_id,  # Add unique ID for frontend
                        'name': model_name,
                        'description': model.get('reasoning', model.get('advantages', '')),
                        'accuracy_estimate': model.get('expected_accuracy', 'Unknown'),
                        'reasoning': model.get('reasoning', ''),
                        'advantages': model.get('advantages', ''),
                        'rank': model.get('rank', i+1)
                    }
                    
                    # Add ALL models to recommended_models (user wants to see all models)
                    recommended_models.append(model_data)
                
                # Keep alternative_models empty to avoid duplicates
                alternative_models = []
                
                recommendations = {
                    'recommended_models': recommended_models,
                    'alternative_models': alternative_models,
                    'scenario_detected': scenario_info,
                    'semantic_analysis': semantic_analysis,
                    'primary_recommendation': primary_recommendation
                }
            else:
                # Fallback to old format if new format not available
                if 'recommended_model' in recs:
                    primary = recs.get('recommended_model')
                    alt = recs.get('alternative_models', [])
                    
                    primary_name = primary.get('name', 'Primary Model') if isinstance(primary, dict) else 'Primary Model'
                    normalized_primary = {
                        'id': f"primary_{primary_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('-', '_')}",
                        'name': primary_name,
                        'description': primary.get('description', '') if isinstance(primary, dict) else str(primary),
                        'accuracy_estimate': 'Unknown',
                        'reasoning': primary.get('reasoning', '') if isinstance(primary, dict) else '',
                        'rank': 1
                    }
                    
                    normalized_alts = []
                    for i, a in enumerate(alt):
                        alt_name = a.get('name', 'Alt Model')
                        normalized_alts.append({
                            'id': f"alt_{i+1}_{alt_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('-', '_')}",
                            'name': alt_name,
                            'description': a.get('description', ''),
                            'accuracy_estimate': 'Unknown',
                            'rank': i + 2
                        })
                    
                    recommendations = {
                        'recommended_models': [normalized_primary],
                        'alternative_models': normalized_alts or backend_fallback['alternative_models'],
                        'scenario_detected': scenario_info,
                        'semantic_analysis': semantic_analysis
                    }
                else:
                    # Update backend fallback to also have unique IDs
                    fallback_primary = backend_fallback['recommended_models'][0]
                    fallback_primary['id'] = 'fallback_random_forest'
                    fallback_primary['rank'] = 1
                    
                    for i, alt_model in enumerate(backend_fallback['alternative_models']):
                        alt_name = alt_model.get('name', f'Alt Model {i+1}')
                        alt_model['id'] = f"fallback_{i+1}_{alt_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('-', '_')}"
                        alt_model['rank'] = i + 2
                    
                    recommendations = backend_fallback
                    
            raw_llm_response = llm_response.get('raw_response', '')

        response_data = {
            'success': True,
            'file_id': file_id,
            'dataset_info': {
                'total_rows': dataset_analysis.get('total_rows', 0),
                'total_columns': dataset_analysis.get('total_columns', 0),
                'numeric_columns': len(dataset_analysis.get('numeric_columns', [])),
                'categorical_columns': len(dataset_analysis.get('categorical_columns', []))
            },
            'user_answers': user_answers,
            'recommendations': recommendations,
            'raw_llm_response': raw_llm_response,
            'warnings': warnings if warnings else None
        }

        # Always return 200 with success true even if fallbacks used; frontend can display warnings
        return jsonify(response_data), 200

    except Exception as e:
        # Final safety net: still return fallback instead of 500
        fallback = {
            'recommended_models': [
                {
                    'name': 'Random Forest',
                    'description': 'Standard fallback model.',
                    'accuracy_estimate': 80,
                    'reasoning': 'Safe default'
                }
            ],
            'alternative_models': []
        }
        return jsonify({
            'success': True,
            'file_id': request.args.get('file_id'),
            'recommendations': fallback,
            'warnings': [f'Critical error in recommendation pipeline: {str(e)}']
        }), 200

@app.route('/api/train', methods=['POST'])
def start_training():
    """
    Start model training using advanced training system
    
    Expected JSON body:
    - file_id: ID of uploaded file
    - model_name: Name of selected model
    
    Returns:
    - JSON response with training results
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
        user_answers = file_info.get('user_answers', {})
        
        # Load the dataset
        import pandas as pd
        df = pd.read_csv(file_path)
        
        # Use advanced training system
        print(f"Starting advanced training for model: {model_name}")
        print(f"Dataset shape: {df.shape}")
        
        result = ml_core.train_advanced_model(
            df=df,
            model_name=model_name,
            target_column=None,  # Will be auto-detected
            file_id=file_id
        )
        
        # Store training info
        file_info['training_completed'] = datetime.now().isoformat()
        file_info['selected_model'] = model_name
        file_info['training_result'] = result
        
        print(f"Training completed successfully: {result}")
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'model_name': model_name,
            'result': result,
            'message': 'Training completed successfully'
        })
    
    except Exception as e:
        print(f"Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Failed to complete training: {str(e)}'
        }), 500

@app.route('/api/training-status/<file_id>', methods=['GET'])
def get_training_status(file_id):
    """
    Get the status of training for a file
    
    Parameters:
    - file_id: File ID for the training
    
    Returns:
    - JSON response with training status and results
    """
    try:
        # Check if file exists
        if file_id not in uploaded_files:
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
        
        file_info = uploaded_files[file_id]
        
        # Check if training was completed
        if 'training_result' in file_info and 'training_completed' in file_info:
            return jsonify({
                'success': True,
                'file_id': file_id,
                'state': 'SUCCESS',
                'progress': 100,
                'status': 'Training completed successfully!',
                'is_complete': True,
                'training_completed': file_info['training_completed'],
                'selected_model': file_info.get('selected_model', ''),
                'results': file_info['training_result']
            })
        else:
            return jsonify({
                'success': True,
                'file_id': file_id,
                'state': 'PENDING',
                'progress': 0,
                'status': 'No training completed for this file',
                'is_complete': False
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get training status: {str(e)}'
        }), 500

@app.route('/api/train-recommended', methods=['POST'])
def train_recommended_model():
    """
    Train the AI-recommended model directly (synchronous)
    
    Expected JSON body:
    - file_id: ID of uploaded file
    
    Returns:
    - JSON response with training results
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'JSON body is required'
            }), 400
        
        file_id = data.get('file_id')
        
        if not file_id:
            return jsonify({
                'success': False,
                'error': 'file_id is required'
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
        
        print(f"\n🚀 TRAINING RECOMMENDED MODEL FOR FILE: {file_id}")
        print(f"📂 File path: {file_path}")
        print(f"👤 User answers: {user_answers}")
        
        # First, get AI recommendations if not already available
        if 'recommendations' not in file_info:
            print("📋 Getting AI recommendations first...")
            dataset_analysis = ml_core.analyze_dataset(file_path)
            llm_response = ml_core.make_llm_request(user_answers, dataset_analysis)
            
            if not llm_response.get('success', False):
                return jsonify({
                    'success': False,
                    'error': 'Failed to get AI recommendations'
                }), 500
            
            file_info['recommendations'] = llm_response.get('recommendations', {})
        
        recommendations = file_info['recommendations']
        
        # Train the recommended model
        training_results = ml_core.train_recommended_model(file_path, recommendations, user_answers)
        
        if training_results.get('success'):
            # Store training results
            file_info['training_results'] = training_results
            file_info['model_trained'] = True
            file_info['training_completed'] = datetime.now().isoformat()
            
            return jsonify({
                'success': True,
                'training_results': training_results,
                'message': 'Model trained successfully!'
            })
        else:
            return jsonify({
                'success': False,
                'error': training_results.get('error', 'Training failed')
            }), 500
    
    except Exception as e:
        print(f"❌ Error in train_recommended_model: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to train model: {str(e)}'
        }), 500

@app.route('/api/train-advanced', methods=['POST'])
def train_advanced_model():
    """
    Train a model using the advanced trainer with 90%+ accuracy optimization
    
    Request JSON:
    {
        "file_id": "uuid",
        "model_name": "Logistic Regression",
        "target_column": "column_name"
    }
    
    Returns:
    - JSON response with training results and model folder information
    """
    try:
        print(f"\n🚀 ADVANCED MODEL TRAINING REQUEST")
        print("="*80)
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        file_id = data.get('file_id')
        model_name = data.get('model_name')
        target_column = data.get('target_column')
        
        # Validate required parameters
        if not file_id:
            return jsonify({
                'success': False,
                'error': 'file_id is required'
            }), 400
        
        if not model_name:
            return jsonify({
                'success': False,
                'error': 'model_name is required'
            }), 400
        
        if not target_column:
            return jsonify({
                'success': False,
                'error': 'target_column is required'
            }), 400
        
        # Check if file exists
        if file_id not in uploaded_files:
            return jsonify({
                'success': False,
                'error': 'File not found. Please upload a file first.'
            }), 404
        
        file_info = uploaded_files[file_id]
        file_path = file_info['file_path']
        
        print(f"📄 Training file: {file_info['original_filename']}")
        print(f"🤖 Model: {model_name}")
        print(f"🎯 Target: {target_column}")
        
        # Validate model availability
        available_models = ml_core.get_available_models()
        if model_name not in available_models:
            return jsonify({
                'success': False,
                'error': f'Model "{model_name}" not available. Available models: {available_models}'
            }), 400
        
        # Train the model using advanced trainer
        training_result = ml_core.train_advanced_model(
            model_name=model_name,
            file_path=file_path,
            target_column=target_column
        )
        
        if training_result['success']:
            print(f"✅ Advanced training completed successfully!")
            
            response_data = {
                'success': True,
                'message': 'Advanced model training completed successfully',
                'model_name': model_name,
                'model_folder': training_result['model_folder'],
                'performance': training_result['performance'],
                'main_score': training_result['main_score'],
                'score_name': training_result['score_name'],
                'problem_type': training_result['problem_type'],
                'threshold_met': training_result['threshold_met'],
                'file_info': {
                    'file_id': file_id,
                    'filename': file_info['original_filename'],
                    'target_column': target_column
                }
            }
            
            if training_result['threshold_met']:
                response_data['message'] += f" - Achieved 90%+ {training_result['score_name']}!"
            else:
                response_data['message'] += f" - {training_result['score_name']}: {training_result['main_score']*100:.1f}%"
            
            return jsonify(response_data)
        
        else:
            print(f"❌ Advanced training failed: {training_result.get('error', 'Unknown error')}")
            return jsonify({
                'success': False,
                'error': training_result.get('error', 'Training failed'),
                'model_name': model_name
            }), 500
    
    except Exception as e:
        print(f"❌ Error in train_advanced_model: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to train advanced model: {str(e)}'
        }), 500

@app.route('/api/available-models', methods=['GET'])
def get_available_models():
    """
    Get list of available models for training
    
    Query parameters:
    - problem_type: 'classification' or 'regression' (optional)
    
    Returns:
    - JSON response with available model names
    """
    try:
        problem_type = request.args.get('problem_type')
        
        available_models = ml_core.get_available_models(problem_type)
        
        return jsonify({
            'success': True,
            'models': available_models,
            'problem_type': problem_type,
            'total_count': len(available_models)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get available models: {str(e)}'
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