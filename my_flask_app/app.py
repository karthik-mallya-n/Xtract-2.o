"""
Flask Application for ML Platform Backend
Main API server handling file uploads, model recommendations, training, and predictions.
"""

import os
import uuid
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import joblib

# Import our custom modules
from core_ml import ml_core
from chart_generator import ChartGenerator
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
    cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001').split(',')
    CORS(app, origins=cors_origins, supports_credentials=True)
    
    # Initialize chart generator
    global chart_generator
    chart_generator = ChartGenerator()
    
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
        
        # Check if file exists in uploaded_files dictionary or find it directly
        file_info = None
        file_path = None
        
        if file_id in uploaded_files:
            # File is in current session
            file_info = uploaded_files[file_id]
            file_path = file_info['file_path']
        else:
            # Try to find the file directly in uploads folder
            import glob
            potential_files = glob.glob(f"{app.config['UPLOAD_FOLDER']}/{file_id}.*")
            if not potential_files:
                # Try pattern matching for UUID-like file IDs
                potential_files = glob.glob(f"{app.config['UPLOAD_FOLDER']}/*{file_id[:8]}*.*")
            
            if potential_files:
                file_path = potential_files[0]
                print(f"Found file directly: {file_path}")
                # Create minimal file_info
                file_info = {
                    'file_id': file_id,
                    'file_path': file_path,
                    'original_filename': os.path.basename(file_path),
                    'user_answers': {}
                }
            else:
                # Try the default Iris dataset for demo purposes
                demo_file = f"{app.config['UPLOAD_FOLDER']}/b0560d95-7006-4035-9ac9-a547229a0071.csv"
                if os.path.exists(demo_file):
                    file_path = demo_file
                    print(f"Using demo file: {file_path}")
                    file_info = {
                        'file_id': file_id,
                        'file_path': file_path,
                        'original_filename': 'iris_demo.csv',
                        'user_answers': {'is_labeled': 'labeled', 'data_type': 'continuous'}
                    }
                else:
                    return jsonify({
                        'success': False,
                        'error': f'File with ID {file_id} not found. Please upload a file first.'
                    }), 404
        user_answers = file_info.get('user_answers', {})
        
        # Load the dataset
        import pandas as pd
        df = pd.read_csv(file_path)
        
        # Try to get target column from user answers or auto-detect
        target_column = None
        
        # Check if this is a clustering model (no target column needed)
        clustering_models = ['kmeans', 'dbscan', 'hierarchical clustering', 'gaussian mixture']
        is_clustering = any(cluster_model in model_name.lower() for cluster_model in clustering_models)
        
        if is_clustering:
            # For clustering, use any column as dummy target (the algorithm will ignore it)
            target_column = df.columns[0]
            print(f"Clustering model detected - using dummy target: {target_column}")
        elif 'target_column' in user_answers:
            target_column = user_answers['target_column']
        else:
            # Auto-detect: assume last column is target for classification/regression
            target_column = df.columns[-1]
        
        # Use advanced training system
        print(f"🚀 Starting advanced training for model: {model_name}")
        print(f"📂 File path: {file_path}")
        print(f"📊 File ID: {file_id}")
        print(f"📄 Original filename: {file_info.get('original_filename', 'Unknown')}")
        
        # Load and validate dataset
        import pandas as pd
        try:
            df = pd.read_csv(file_path)
            print(f"📈 Dataset shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return jsonify({
                'success': False,
                'error': f'Failed to load dataset: {str(e)}'
            }), 500

        user_answers = file_info.get('user_answers', {})
        
        # Try to get target column from user answers or auto-detect
        target_column = None
        
        # Check if this is a clustering model (no target column needed)
        clustering_models = ['kmeans', 'dbscan', 'hierarchical clustering', 'gaussian mixture']
        is_clustering = any(cluster_model in model_name.lower() for cluster_model in clustering_models)
        
        if is_clustering:
            # For clustering, use any column as dummy target (the algorithm will ignore it)
            target_column = df.columns[0]
            print(f"Clustering model detected - using dummy target: {target_column}")
        elif 'target_column' in user_answers:
            target_column = user_answers['target_column']
        else:
            # Auto-detect: assume last column is target for classification/regression
            target_column = df.columns[-1]
        
        print(f"🎯 Target column: {target_column}")
        
        # Train the model
        result = ml_core.train_advanced_model(
            model_name=model_name,
            file_path=file_path,
            target_column=target_column
        )
        
        # Store training info
        file_info['training_completed'] = datetime.now().isoformat()
        file_info['selected_model'] = model_name
        file_info['training_result'] = result
        
        print(f"Training completed successfully: {result}")
        
        # Get feature information from the saved model for frontend display
        feature_info = {}
        if result['success'] and 'model_folder' in result:
            try:
                # Check for metadata.json in the model folder
                metadata_path = f"{result['model_folder']}/metadata.json"
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    feature_info = {
                        'feature_names': metadata.get('feature_names', []),
                        'target_column': metadata.get('target_column', target_column),
                        'problem_type': metadata.get('problem_type', 'classification')
                    }
                    print(f"📊 Feature info loaded from metadata: {len(feature_info.get('feature_names', []))} features")
                else:
                    # Try legacy .joblib file format
                    model_path = f"{result['model_folder']}.joblib"
                    if os.path.exists(model_path):
                        saved_model_data = joblib.load(model_path)
                        feature_info = {
                            'feature_names': saved_model_data.get('feature_names', []),
                            'target_column': saved_model_data.get('target_column', target_column),
                            'problem_type': saved_model_data.get('problem_type', 'classification')
                        }
                        print(f"📊 Feature info loaded from joblib: {len(feature_info.get('feature_names', []))} features")
            except Exception as e:
                print(f"⚠️ Could not load feature info: {e}")
                # Fallback: extract feature names from the dataset
                try:
                    import pandas as pd
                    temp_df = pd.read_csv(file_path)
                    feature_names = [col for col in temp_df.columns if col != target_column]
                    feature_info = {
                        'feature_names': feature_names,
                        'target_column': target_column,
                        'problem_type': 'classification' if temp_df[target_column].nunique() <= 20 else 'regression'
                    }
                    print(f"📊 Feature info from dataset: {len(feature_names)} features")
                except Exception as inner_e:
                    print(f"❌ Could not extract features: {inner_e}")
                    feature_info = {'feature_names': [], 'target_column': target_column, 'problem_type': 'classification'}
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'model_name': model_name,
            'result': result,
            'feature_info': feature_info,
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

@app.route('/api/generate-training-script', methods=['POST'])
def generate_training_script():
    """
    Generate a complete high-accuracy Python training script using Pipeline + GridSearchCV
    
    Request JSON:
    {
        "file_id": "uploaded_file_id",
        "model_name": "Random Forest Classifier", 
        "target_column": "loan_approved",
        "columns_to_drop": ["customer_id", "timestamp"] (optional),
        "scoring_metric": "accuracy" (optional)
    }
    
    Returns:
    - JSON response with complete Python script
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        file_id = data.get('file_id')
        model_name = data.get('model_name')
        target_column = data.get('target_column')
        columns_to_drop = data.get('columns_to_drop', [])
        scoring_metric = data.get('scoring_metric')
        
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
        
        # Check if file exists
        if file_id not in uploaded_files:
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
        
        file_info = uploaded_files[file_id]
        file_path = file_info['path']
        
        print(f"\n🚀 GENERATING HIGH-ACCURACY TRAINING SCRIPT")
        print("="*80)
        print(f"🤖 Model: {model_name}")
        print(f"📄 Dataset: {file_path}")
        print(f"🎯 Target: {target_column}")
        print(f"🗑️ Drop columns: {columns_to_drop}")
        print(f"📊 Scoring metric: {scoring_metric}")
        
        # Generate the training script
        script_result = ml_core.generate_high_accuracy_training_script(
            model_name=model_name,
            file_path=file_path, 
            target_column=target_column,
            columns_to_drop=columns_to_drop,
            scoring_metric=scoring_metric
        )
        
        if script_result['success']:
            print(f"✅ Script generation completed successfully!")
            
            return jsonify({
                'success': True,
                'message': 'High-accuracy training script generated successfully',
                'script': script_result['script'],
                'model_info': script_result['model_info'],
                'scenario_type': script_result['scenario_type'],
                'file_info': {
                    'file_id': file_id,
                    'filename': file_info['original_filename'],
                    'file_path': script_result['file_path'],
                    'target_column': script_result['target_column']
                },
                'configuration': {
                    'model_name': model_name,
                    'columns_to_drop': columns_to_drop,
                    'scoring_metric': script_result['scoring_metric']
                }
            })
        
        else:
            print(f"❌ Script generation failed: {script_result.get('error', 'Unknown error')}")
            return jsonify({
                'success': False,
                'error': script_result.get('error', 'Script generation failed'),
                'model_name': model_name
            }), 500
            
    except Exception as e:
        print(f"❌ Error in generate_training_script: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to generate training script: {str(e)}'
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
        
        # Find the most recent model folder for this file_id (new pipeline format)
        import glob
        import os
        
        # First try to find models using the new folder format
        model_folders = glob.glob(f"{app.config['MODEL_STORAGE_PATH']}/*_*")
        model_folders = [f for f in model_folders if os.path.isdir(f)]
        
        if not model_folders:
            return jsonify({
                'success': False,
                'error': 'No trained models found. Please train a model first.'
            }), 404
        
        # Sort by timestamp (most recent first) - extract timestamp from folder name
        def get_timestamp(folder_path):
            folder_name = os.path.basename(folder_path)
            parts = folder_name.split('_')
            if len(parts) >= 2:
                # Check if last part looks like a timestamp (YYYYMMDD_HHMMSS)
                timestamp_part = parts[-1]
                if len(timestamp_part) == 6 and timestamp_part.isdigit():
                    # Get the date part too (second to last)
                    if len(parts) >= 3 and len(parts[-2]) == 8 and parts[-2].isdigit():
                        return f"{parts[-2]}_{timestamp_part}"  # YYYYMMDD_HHMMSS format
                    else:
                        return f"20251115_{timestamp_part}"  # Default to today's date
                return "00000000_000000"  # Invalid format
            return "00000000_000000"  # No timestamp
        
        # Filter out old models without proper timestamps
        timestamped_models = []
        for folder in model_folders:
            ts = get_timestamp(folder)
            if ts != "00000000_000000":
                timestamped_models.append(folder)
        
        if not timestamped_models:
            return jsonify({
                'success': False,
                'error': 'No valid trained models found. Please train a model first.'
            }), 404
        
        timestamped_models.sort(key=get_timestamp, reverse=True)
        model_folder = timestamped_models[0]
        
        print(f"🔍 Using model folder: {model_folder}")
        
        # Load metadata to get feature information
        metadata_path = os.path.join(model_folder, 'metadata.json')
        if not os.path.exists(metadata_path):
            return jsonify({
                'success': False,
                'error': 'Model metadata not found. Model may be corrupted.'
            }), 500
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        feature_names = metadata['feature_names']
        target_column = metadata['target_column']
        problem_type = metadata['problem_type']
        
        print(f"📊 Features: {feature_names}")
        print(f"🎯 Target: {target_column}")
        print(f"📈 Problem type: {problem_type}")
        
        # Prepare input data for prediction
        import pandas as pd
        import numpy as np
        
        # Create DataFrame from input data with proper feature names
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        for feature in feature_names:
            if feature not in input_df.columns:
                print(f"⚠️  Missing feature '{feature}', setting to 0")
                input_df[feature] = 0
        
        # Select and order features correctly (excluding target column)
        features_for_prediction = [f for f in feature_names if f != target_column]
        input_df = input_df[features_for_prediction]
        
        print(f"🔄 Input data shape: {input_df.shape}")
        print(f"🔄 Input features: {list(input_df.columns)}")
        
        # Use direct model loading (bypass advanced_model_trainer for compatibility)
        try:
            # Load the pipeline model directly
            model_path = os.path.join(model_folder, 'model.pkl')
            if not os.path.exists(model_path):
                return jsonify({
                    'success': False,
                    'error': 'Model file not found. Model may be corrupted.'
                }), 500
            
            # Load the trained pipeline
            import joblib
            pipeline = joblib.load(model_path)
            
            print(f"✅ Pipeline loaded successfully: {type(pipeline)}")
            
            # Make prediction using the pipeline directly
            predictions = pipeline.predict(input_df)
            prediction = predictions[0] if len(predictions) > 0 else 0
            
            print(f"✅ Prediction: {prediction}")
            
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Prediction failed: {str(e)}'
            }), 500
        
        # For classification, prediction is already the class label
        # For regression, prediction is the numerical value
        prediction_decoded = prediction
        
        return jsonify({
            'success': True,
            'prediction': str(prediction_decoded),
            'raw_prediction': float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
            'confidence': None,  # Confidence not available in new system yet
            'probabilities': None,  # Probabilities not available in new system yet
            'model_info': {
                'model_name': metadata.get('model_name', 'Unknown'),
                'training_date': metadata.get('timestamp', ''),
                'accuracy': metadata.get('performance', {}).get('accuracy', metadata.get('performance', {}).get('r2_score', 0))
            },
            'input_data': input_data,
            'feature_info': {
                'feature_names': features_for_prediction,
                'target_column': target_column,
                'problem_type': problem_type
            }
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

@app.route('/api/dataset/<file_id>', methods=['GET'])
def get_dataset_info(file_id):
    """
    Get dataset information and sample data for visualization
    
    Returns:
    - JSON response with dataset columns, sample data, and metadata
    """
    try:
        # Check if file exists
        if file_id not in uploaded_files:
            return jsonify({
                'success': False,
                'error': 'File not found. Please upload a file first.'
            }), 404

        file_info = uploaded_files[file_id]
        file_path = file_info['file_path']

        # Load and analyze the dataset
        import pandas as pd
        
        # Read the file based on extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif ext.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif ext.lower() == '.json':
            df = pd.read_json(file_path)
        else:
            return jsonify({
                'success': False,
                'error': 'Unsupported file format'
            }), 400

        # Get basic dataset information
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Get sample data (first 10 rows)
        sample_data = df.head(10).to_dict('records')
        
        # Convert numpy types to Python types for JSON serialization
        for row in sample_data:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None
                elif hasattr(value, 'item'):  # numpy types
                    row[key] = value.item()

        return jsonify({
            'success': True,
            'file_id': file_id,
            'filename': file_info.get('filename', ''),
            'dataset_info': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns': df.columns.tolist(),
                'numeric_columns': numeric_columns,
                'categorical_columns': categorical_columns,
                'datetime_columns': datetime_columns,
                'sample_data': sample_data,
                'data_types': df.dtypes.astype(str).to_dict()
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get dataset info: {str(e)}'
        }), 500

@app.route('/api/generate-chart', methods=['POST'])
def generate_chart():
    """
    Generate a chart based on uploaded data and configuration
    
    Expected JSON body:
    - file_id: ID of uploaded file
    - chart_type: Type of chart to generate
    - config: Chart configuration
    - export_format: Output format (html, png, svg, pdf)
    
    Returns:
    - JSON response with chart data
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'JSON body is required'
            }), 400
        
        file_id = data.get('file_id')
        chart_type = data.get('chart_type')
        config = data.get('config', {})
        export_format = data.get('export_format', 'html')
        
        if not file_id:
            return jsonify({
                'success': False,
                'error': 'file_id is required'
            }), 400
        
        if not chart_type:
            return jsonify({
                'success': False,
                'error': 'chart_type is required'
            }), 400
        
        # Check if file exists
        if file_id not in uploaded_files:
            return jsonify({
                'success': False,
                'error': 'File not found. Please upload a file first.'
            }), 404
        
        file_info = uploaded_files[file_id]
        file_path = file_info['file_path']
        
        # Load the dataset
        import pandas as pd
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                return jsonify({
                    'success': False,
                    'error': 'Unsupported file format'
                }), 400
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to load dataset: {str(e)}'
            }), 500
        
        # Generate the chart
        print(f"🎨 Generating {chart_type} chart for file {file_id}")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"⚙️ Config: {config}")
        
        result = chart_generator.generate_chart(df, chart_type, config, export_format)
        
        if result['success']:
            print(f"✅ Chart generated successfully")
            return jsonify({
                'success': True,
                'chart_type': chart_type,
                'config': config,
                'html_content': result.get('html_content'),
                'image_url': result.get('image_url'),
                'format': result.get('format'),
                'warning': result.get('warning')
            })
        else:
            print(f"❌ Chart generation failed: {result['error']}")
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
    
    except Exception as e:
        print(f"❌ Error in generate_chart: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to generate chart: {str(e)}'
        }), 500

@app.route('/api/export-chart', methods=['POST'])
def export_chart():
    """
    Export a chart in specified format
    
    Expected JSON body:
    - file_id: ID of uploaded file
    - chart_type: Type of chart to generate
    - config: Chart configuration
    - export_format: Output format (png, svg, pdf, html)
    
    Returns:
    - File download or JSON response
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'JSON body is required'
            }), 400
        
        file_id = data.get('file_id')
        chart_type = data.get('chart_type')
        config = data.get('config', {})
        export_format = data.get('export_format', 'png')
        
        if not file_id or not chart_type:
            return jsonify({
                'success': False,
                'error': 'file_id and chart_type are required'
            }), 400
        
        # Check if file exists
        if file_id not in uploaded_files:
            return jsonify({
                'success': False,
                'error': 'File not found. Please upload a file first.'
            }), 404
        
        file_info = uploaded_files[file_id]
        file_path = file_info['file_path']
        
        # Load the dataset
        import pandas as pd
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                return jsonify({
                    'success': False,
                    'error': 'Unsupported file format'
                }), 400
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to load dataset: {str(e)}'
            }), 500
        
        # Generate the chart for export
        print(f"📤 Exporting {chart_type} chart as {export_format}")
        
        result = chart_generator.generate_chart(df, chart_type, config, export_format)
        
        if not result['success']:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
        
        # Return file or content based on format
        if export_format == 'html':
            # Return HTML content as downloadable file
            from io import BytesIO
            html_content = result.get('html_content', '')
            
            buffer = BytesIO()
            buffer.write(html_content.encode('utf-8'))
            buffer.seek(0)
            
            return send_file(
                buffer,
                as_attachment=True,
                download_name=f'chart_{chart_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html',
                mimetype='text/html'
            )
        
        else:
            # Return image file
            file_path_result = result.get('file_path')
            filename = result.get('filename')
            
            if file_path_result and os.path.exists(file_path_result):
                return send_file(
                    file_path_result,
                    as_attachment=True,
                    download_name=filename
                )
            else:
                return jsonify({
                    'success': False,
                    'error': 'Export file not found'
                }), 500
    
    except Exception as e:
        print(f"❌ Error in export_chart: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to export chart: {str(e)}'
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
    print(f"CORS origins: {os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001')}")
    
    # Disable reloader to prevent duplicate execution in debug mode
    app.run(host='0.0.0.0', port=port, debug=debug_mode, use_reloader=False)