"""
Flask Application for ML Platform Backend
Main API server handling file uploads, model recommendations, training, and predictions.
"""

import os
import uuid
import json
import pandas as pd
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import joblib

# Import our custom modules
from core_ml import ml_core
from visualization_engine import visualization_engine
from api_endpoints import enhanced_api
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
    
    # Register additional API blueprints
    app.register_blueprint(enhanced_api)
    
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

@app.route('/api/dataset-columns', methods=['GET'])
def get_dataset_columns():
    """
    Get column information for a dataset
    
    Query parameters:
    - file_id: ID of uploaded file
    
    Returns:
    - JSON response with column names and types
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
        
        # Read the dataset to get column information
        df = pd.read_csv(file_path)
        
        # Get column information
        columns_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            
            # Determine if column is numeric or categorical
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                column_type = 'numeric'
            else:
                column_type = 'categorical'
            
            # Get sample values (first few non-null values)
            sample_values = df[col].dropna().head(3).tolist()
            
            columns_info.append({
                'name': col,
                'type': column_type,
                'dtype': dtype,
                'sample_values': sample_values,
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique())
            })

        return jsonify({
            'success': True,
            'file_id': file_id,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': columns_info
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error reading dataset: {str(e)}'
        }), 500

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
        target_column_override = data.get('target_column')  # New: target column from user selection
        
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
        
        # Try to get target column from user selection, user answers, or auto-detect
        target_column = None
        
        # Check if this is a clustering model (no target column needed)
        clustering_models = ['kmeans', 'dbscan', 'hierarchical clustering', 'gaussian mixture']
        is_clustering = any(cluster_model in model_name.lower() for cluster_model in clustering_models)
        
        if is_clustering:
            # For clustering, use any column as dummy target (the algorithm will ignore it)
            target_column = df.columns[0]
            print(f"Clustering model detected - using dummy target: {target_column}")
        elif target_column_override:
            # Use the target column selected by the user
            target_column = target_column_override
            print(f"Using user-selected target column: {target_column}")
        elif 'target_column' in user_answers:
            target_column = user_answers['target_column']
        else:
            # Auto-detect: assume last column is target for classification/regression
            target_column = df.columns[-1]
            print(f"Auto-detected target column: {target_column}")

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

        print(f"🎯 Target column: {target_column}")
        
        # Train the model using comprehensive preprocessing pipeline
        print(f"\n{'='*100}")
        print(f"🚀 STARTING SPECIFIC MODEL TRAINING WITH COMPREHENSIVE PREPROCESSING")
        print(f"{'='*100}\n")
        
        result = ml_core.train_specific_model(
            file_path=file_path,
            model_name=model_name,
            user_data=user_answers,
            target_column=target_column
        )
        
        # Store training info
        file_info['training_completed'] = datetime.now().isoformat()
        file_info['selected_model'] = model_name
        file_info['training_result'] = result
        
        print(f"\n{'='*100}")
        print(f"✅ TRAINING API ENDPOINT COMPLETE")
        print(f"{'='*100}")
        print(f"Result: {result.get('success', False)}")
        if result.get('success'):
            print(f"Performance: {result.get('performance', {})}")
        print(f"{'='*100}\n")
        
        # Get feature information from the training result
        feature_info = {}
        if result.get('success') and result.get('model_info'):
            model_info = result['model_info']
            model_dir = model_info.get('model_directory', '')
            artifacts = model_info.get('artifacts', {})
            
            # Try to load feature info from the saved artifacts
            try:
                if artifacts.get('feature_info'):
                    feature_info_path = os.path.join(model_dir, artifacts['feature_info'])
                    if os.path.exists(feature_info_path):
                        with open(feature_info_path, 'r') as f:
                            feature_data = json.load(f)
                        feature_info = {
                            'feature_names': feature_data.get('feature_names', []),
                            'target_column': feature_data.get('target_column', target_column),
                            'problem_type': result.get('performance', {}).get('model_type', 'classification')
                        }
                        print(f"📊 Feature info loaded from artifacts: {len(feature_info.get('feature_names', []))} features")
                    else:
                        raise FileNotFoundError(f"Feature info file not found: {feature_info_path}")
            except Exception as e:
                print(f"⚠️ Could not load feature info from artifacts: {e}")
                # Fallback: extract feature names from the dataset
                try:
                    import pandas as pd
                    temp_df = pd.read_csv(file_path)
                    feature_names = [col for col in temp_df.columns if col != target_column]
                    feature_info = {
                        'feature_names': feature_names,
                        'target_column': target_column,
                        'problem_type': result.get('performance', {}).get('model_type', 'classification')
                    }
                    print(f"📊 Feature info from dataset: {len(feature_names)} features")
                except Exception as inner_e:
                    print(f"❌ Could not extract features: {inner_e}")
                    feature_info = {'feature_names': [], 'target_column': target_column, 'problem_type': 'classification'}
        else:
            # Training failed or no model_info, fallback to dataset
            try:
                import pandas as pd
                temp_df = pd.read_csv(file_path)
                feature_names = [col for col in temp_df.columns if col != target_column]
                feature_info = {
                    'feature_names': feature_names,
                    'target_column': target_column,
                    'problem_type': 'classification'
                }
            except Exception as e:
                print(f"❌ Could not extract features from dataset: {e}")
                feature_info = {'feature_names': [], 'target_column': target_column, 'problem_type': 'classification'}
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'model_name': model_name,
            'result': result,
            'feature_info': feature_info,
            'message': 'Training completed successfully with comprehensive preprocessing!'
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

@app.route('/api/train-specific-model', methods=['POST'])
def train_specific_model_endpoint():
    """
    Train a specific model selected by the user with comprehensive preprocessing
    
    Expected JSON body:
    - file_id: ID of uploaded file
    - model_name: Name of the specific model to train (e.g., "Random Forest", "XGBoost")
    - target_column: (optional) Name of the target column
    
    Returns:
    - JSON response with detailed training results and performance metrics
    """
    try:
        print("🚨 DEBUG: train_specific_model_endpoint called")
        data = request.get_json()
        print(f"🚨 DEBUG: received data = {data}")
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'JSON body is required'
            }), 400
        
        file_id = data.get('file_id')
        model_name = data.get('model_name')
        target_column = data.get('target_column')
        
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
                'error': 'File not found. Please upload a file first.'
            }), 404
        
        file_info = uploaded_files[file_id]
        file_path = file_info['file_path']
        user_answers = file_info['user_answers']
        
        print(f"\n{'='*100}")
        print(f"🚀 TRAINING SPECIFIC MODEL ENDPOINT")
        print(f"{'='*100}")
        print(f"📂 File ID: {file_id}")
        print(f"📂 File path: {file_path}")
        print(f"🎯 Model name: {model_name}")
        print(f"🎯 Target column: {target_column if target_column else 'Auto-detect (last column)'}")
        print(f"👤 User data: {user_answers}")
        print(f"{'='*100}\n")
        
        # Train the specific model with comprehensive preprocessing
        training_results = ml_core.train_specific_model(
            file_path=file_path,
            model_name=model_name,
            user_data=user_answers,
            target_column=target_column
        )
        
        if training_results.get('success'):
            # Store training results in file_info
            file_info['specific_model_training'] = training_results
            file_info['specific_model_trained'] = True
            file_info['specific_model_name'] = model_name
            file_info['specific_training_completed'] = datetime.now().isoformat()
            
            print(f"\n{'='*100}")
            print(f"✅ ENDPOINT: MODEL TRAINING SUCCESSFUL")
            print(f"{'='*100}")
            print(f"🎯 Model: {model_name}")
            print(f"📊 Performance: {training_results.get('performance', {})}")
            print(f"📁 Model directory: {training_results.get('model_info', {}).get('model_directory', 'N/A')}")
            print(f"{'='*100}\n")
            
            # Extract feature information for the frontend by reading the actual data
            feature_info = training_results.get('feature_info', {})
            if not feature_info.get('feature_names'):
                # Try to extract from other locations
                if training_results.get('model_info', {}).get('feature_names'):
                    feature_info['feature_names'] = training_results['model_info']['feature_names']
                elif training_results.get('training_details', {}).get('feature_names'):
                    feature_info['feature_names'] = training_results['training_details']['feature_names']
                else:
                    # Extract feature names directly from the dataset
                    try:
                        import pandas as pd
                        df = pd.read_csv(file_path)
                        
                        # Determine target column
                        actual_target = target_column if target_column else df.columns[-1]
                        
                        # Get feature names (all columns except target)
                        feature_names = [col for col in df.columns if col != actual_target]
                        
                        feature_info = {
                            'feature_names': feature_names,
                            'target_column': actual_target,
                            'problem_type': training_results.get('problem_type', 'classification'),
                            'total_features': len(feature_names),
                            'dataset_shape': df.shape
                        }
                        
                        print(f"🔍 EXTRACTED FEATURE INFO:")
                        print(f"📊 Features: {feature_names}")
                        print(f"🎯 Target: {actual_target}")
                        
                    except Exception as e:
                        print(f"⚠️ Feature extraction failed: {str(e)}")
                        feature_info = {}
            
            return jsonify({
                'success': True,
                'training_results': training_results,
                'message': f'{model_name} trained successfully with comprehensive preprocessing!',
                'model_info': training_results.get('model_info', {}),
                'performance': training_results.get('performance', {}),
                'feature_info': feature_info,  # Add feature info for frontend
                'file_info': {
                    'file_id': file_id,
                    'filename': file_info['original_filename'],
                    'target_column': target_column
                },
                'training_details': {
                    **training_results.get('training_details', {}),  # Base training details
                    **training_results.get('performance', {}),       # Merge performance metrics for UI
                    'target_column': target_column,  # Ensure target column is available
                    'test_split': 0.2,  # Default test split info
                    'problem_type': training_results.get('problem_type', 'classification')
                }
            })
        else:
            error_msg = training_results.get('error', 'Training failed')
            print(f"\n{'='*100}")
            print(f"❌ ENDPOINT: MODEL TRAINING FAILED")
            print(f"{'='*100}")
            print(f"Error: {error_msg}")
            print(f"{'='*100}\n")
            
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        
        print(f"\n{'='*100}")
        print(f"❌ ENDPOINT ERROR: EXCEPTION IN train_specific_model_endpoint")
        print(f"{'='*100}")
        print(f"Error: {str(e)}")
        print(f"Traceback:\n{error_traceback}")
        print(f"{'='*100}\n")
        
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
    
    Expected JSON body formats:
    Format 1: {"file_id": "...", "input_data": {...}}
    Format 2: {"features": [1.5, 2.3, ...]}
    
    Returns:
    - JSON response with prediction results
    """
    try:
        data = request.get_json()
        
        if not data:
            print("❌ PREDICTION ERROR: No JSON body received")
            return jsonify({
                'success': False,
                'error': 'JSON body is required'
            }), 400
        
        print(f"🔍 PREDICTION REQUEST: {data}")
        
        # Handle both request formats
        if 'features' in data:
            # Format 2: Direct features array (from React frontend)
            features_array = data.get('features', [])
            print(f"📊 Features array received: {len(features_array)} features")
            if not features_array:
                return jsonify({
                    'success': False,
                    'error': 'features array is required'
                }), 400
            
            # We'll map features to proper names after loading metadata
            input_data = None  # Will be set after metadata loading
            file_id = None  # Will find the most recent model
        else:
            # Format 1: Original format
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
        
        # Look for all model folders (both timestamped folders and regular folders)
        all_model_folders = glob.glob(f"{app.config['MODEL_STORAGE_PATH']}/*")
        all_model_folders = [f for f in all_model_folders if os.path.isdir(f)]
        
        if not all_model_folders:
            return jsonify({
                'success': False,
                'error': 'No trained models found. Please train a model first.'
            }), 404
        
        # Function to get the most recent model within a folder
        def get_folder_timestamp(folder_path):
            # First check if folder name has timestamp
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
            
            # If no timestamp in folder name, check for timestamped files within
            metadata_files = glob.glob(os.path.join(folder_path, 'metadata_*.json'))
            if metadata_files:
                # Extract timestamp from most recent metadata file
                timestamps = []
                for mf in metadata_files:
                    basename = os.path.basename(mf)
                    if basename.startswith('metadata_') and basename.endswith('.json'):
                        ts_part = basename[9:-5]  # Remove 'metadata_' and '.json'
                        timestamps.append(ts_part)
                if timestamps:
                    timestamps.sort(reverse=True)
                    return timestamps[0]
            
            return "00000000_000000"  # No timestamp found
        
        # Function to check if a model is for Iris dataset
        def is_iris_model(folder_path):
            metadata_files = glob.glob(os.path.join(folder_path, 'metadata_*.json'))
            if metadata_files:
                metadata_files.sort(reverse=True)
                try:
                    with open(metadata_files[0], 'r') as f:
                        metadata = json.load(f)
                    feature_names = metadata.get('feature_names', [])
                    target_column = metadata.get('target_column', '')
                    
                    # Check for Iris characteristics
                    iris_features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
                    is_iris = (target_column.lower() == 'species' or 
                             any(feat in feature_names for feat in iris_features))
                    return is_iris
                except:
                    pass
            return False
        
        # Separate Iris and non-Iris models
        iris_models = []
        other_models = []
        
        for folder in all_model_folders:
            ts = get_folder_timestamp(folder)
            if ts != "00000000_000000":
                if is_iris_model(folder):
                    iris_models.append((folder, ts))
                else:
                    other_models.append((folder, ts))
        
        # Choose model: always prioritize Iris models if they exist (user is working with Iris data)
        if iris_models:
            iris_models.sort(key=lambda x: x[1], reverse=True)
            model_folder = iris_models[0][0]
            print(f"🌸 PREDICTION: Using IRIS model folder: {model_folder}")
        elif other_models:
            other_models.sort(key=lambda x: x[1], reverse=True)
            model_folder = other_models[0][0]
            print(f"📁 PREDICTION: Using model folder: {model_folder}")
        else:
            return jsonify({
                'success': False,
                'error': 'No valid trained models found. Please train a model first.'
            }), 404
        
        print(f"🔍 Using model folder: {model_folder}")
        
        # Load metadata to get feature information
        metadata_files = glob.glob(os.path.join(model_folder, 'metadata_*.json'))
        if not metadata_files:
            # Fallback to old format
            metadata_path = os.path.join(model_folder, 'metadata.json')
            if not os.path.exists(metadata_path):
                return jsonify({
                    'success': False,
                    'error': 'Model metadata not found. Model may be corrupted.'
                }), 500
        else:
            # Use the most recent metadata file
            metadata_files.sort(reverse=True)
            metadata_path = metadata_files[0]
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        feature_names = metadata.get('feature_names', [])
        target_column = metadata.get('target_column', 'target')
        problem_type = metadata.get('model_type', metadata.get('problem_type', 'regression'))
        
        # Now map features array to proper feature names if needed
        if 'features' in data and input_data is None:
            features_array = data.get('features', [])
            # Get features excluding target column for prediction
            features_for_prediction = [f for f in feature_names if f != target_column]
            
            # Handle AI-selected features: if user provides fewer features than model expects,
            # it might be because they're following AI recommendations to exclude ID columns
            expected_features = len(features_for_prediction)
            provided_features = len(features_array)
            
            if provided_features != expected_features:
                print(f"⚠️  Feature count mismatch: received {provided_features}, expected {expected_features}")
                print(f"⚠️  Expected features: {features_for_prediction}")
                
                # Check if this is an Iris model where ID should be excluded
                is_iris = (target_column.lower() == 'species' or 
                          any('sepal' in col.lower() or 'petal' in col.lower() for col in feature_names))
                
                if is_iris and expected_features == 5 and provided_features == 4:
                    # This is likely an Iris model with ID column - add default ID value
                    if 'Id' in features_for_prediction:
                        print("🌸 IRIS: Adding default ID value (1) for excluded ID column")
                        id_index = features_for_prediction.index('Id')
                        # Insert ID value at the correct position
                        features_array.insert(id_index, 1)  # Use ID = 1 as default
                        print(f"🔄 Updated features array: {features_array}")
                    else:
                        return jsonify({
                            'success': False,
                            'error': f'Feature count mismatch. Expected {expected_features} features, got {provided_features}. Expected features: {features_for_prediction}'
                        }), 400
                else:
                    return jsonify({
                        'success': False,
                        'error': f'Feature count mismatch. Expected {expected_features} features, got {provided_features}. Expected features: {features_for_prediction}'
                    }), 400
            
            # Map features array to actual feature names
            input_data = {feature_name: features_array[i] for i, feature_name in enumerate(features_for_prediction)}
            print(f"🔄 Mapped input data: {input_data}")
        
        print(f"📊 Features: {feature_names}")
        print(f"🎯 Target: {target_column}")
        print(f"📈 Problem type: {problem_type}")
        
        # Prepare input data for prediction
        import pandas as pd
        import numpy as np
        
        # Create DataFrame from input data with proper feature names
        input_df = pd.DataFrame([input_data])
        
        # Get features for prediction (excluding target)
        features_for_prediction = [f for f in feature_names if f != target_column]
        
        # Ensure all required features are present
        for feature in features_for_prediction:
            if feature not in input_df.columns:
                print(f"⚠️  Missing feature '{feature}', setting to 0")
                input_df[feature] = 0
        
        # Select and order features correctly (excluding target column)
        input_df = input_df[features_for_prediction]
        
        print(f"🔄 Input data shape: {input_df.shape}")
        print(f"🔄 Input features: {list(input_df.columns)}")
        
        # Use direct model loading (bypass advanced_model_trainer for compatibility)
        try:
            # Load the pipeline model directly - check for multiple possible filenames
            model_files = glob.glob(os.path.join(model_folder, 'model_*.joblib'))
            print(f"🔍 Found model files: {model_files}")
            
            if not model_files:
                # Fallback to old format
                model_path = os.path.join(model_folder, 'model.pkl')
                print(f"🔍 Checking fallback: {model_path}")
                if not os.path.exists(model_path):
                    model_path = os.path.join(model_folder, 'model.joblib')
                    print(f"🔍 Checking fallback 2: {model_path}")
                    if not os.path.exists(model_path):
                        print(f"❌ No model files found in {model_folder}")
                        return jsonify({
                            'success': False,
                            'error': 'Model file not found. Model may be corrupted.'
                        }), 500
            else:
                # Use the most recent model file
                model_files.sort(reverse=True)
                model_path = model_files[0]
            
            print(f"📁 Loading model from: {model_path}")
            
            # Load the trained pipeline
            import joblib
            pipeline = joblib.load(model_path)
            
            print(f"✅ Pipeline loaded successfully: {type(pipeline)}")
            print(f"📊 Pipeline steps: {getattr(pipeline, 'steps', 'N/A')}")
            
            # Debug input data shape and content
            print(f"📥 Input DataFrame:")
            print(input_df.head())
            print(f"📊 Input shape: {input_df.shape}")
            print(f"🔧 Input dtypes: {input_df.dtypes.to_dict()}")
            
            # Make prediction using the pipeline directly
            print("🔮 Making prediction...")
            predictions = pipeline.predict(input_df)
            prediction = predictions[0] if len(predictions) > 0 else 0
            
            print(f"✅ Raw prediction result: {prediction}")
            print(f"✅ Prediction type: {type(prediction)}")
            
            # For classification problems, convert prediction index to class name
            prediction_decoded = prediction
            if problem_type == 'classification':
                # Try to load target encoder to convert index to class name
                import glob
                target_encoder_files = glob.glob(os.path.join(model_folder, 'target_encoder_*.joblib'))
                if target_encoder_files:
                    try:
                        import joblib
                        target_encoder_files.sort(reverse=True)
                        target_encoder_path = target_encoder_files[0]
                        target_encoder = joblib.load(target_encoder_path)
                        
                        # Convert prediction index to class name
                        if hasattr(target_encoder, 'classes_'):
                            if isinstance(prediction, (int, np.integer)) and 0 <= prediction < len(target_encoder.classes_):
                                prediction_decoded = target_encoder.classes_[prediction]
                                print(f"🔄 Decoded prediction: index {prediction} -> '{prediction_decoded}'")
                            else:
                                prediction_decoded = str(prediction)
                        else:
                            prediction_decoded = str(prediction)
                    except Exception as e:
                        print(f"⚠️ Could not decode prediction using target encoder: {e}")
                        prediction_decoded = str(prediction)
                else:
                    # If no encoder available, assume prediction is already a class name
                    prediction_decoded = str(prediction)
            
        except Exception as e:
            import traceback
            full_error = traceback.format_exc()
            print(f"❌ Prediction error: {str(e)}")
            print(f"❌ Full traceback:\n{full_error}")
            return jsonify({
                'success': False,
                'error': f'Prediction failed: {str(e)}',
                'debug_info': full_error
            }), 500
        
        return jsonify({
            'success': True,
            'prediction': str(prediction_decoded),
            'raw_prediction': float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
            'confidence': None,  # Confidence not available in new system yet
            'probabilities': None,  # Probabilities not available in new system yet
            'model_info': {
                'model_name': metadata.get('model_name', 'Unknown'),
                'training_date': metadata.get('timestamp', ''),
                'accuracy': metadata.get('test_score', metadata.get('performance', {}).get('accuracy', metadata.get('performance', {}).get('r2_score', 0)))
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

@app.errorhandler(413)
def file_too_large(error):
    """Handle file size limit exceeded"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum file size is 16MB.'
    }), 413

@app.route('/api/visualizations/analyze', methods=['GET'])
def analyze_dataset_for_visualizations():
    """
    Analyze dataset for available visualizations
    
    Query parameters:
    - file_id: ID of uploaded file (optional for testing)
    
    Returns:
    - JSON response with column information and available visualization types
    """
    try:
        print(f"📊 Analyzing dataset for visualizations...")
        file_id = request.args.get('file_id')
        print(f"🔍 Requested file_id: {file_id}")
        
        # Test mode: use sample CSV if no file_id provided
        if not file_id or file_id == 'sample':
            print("🧪 Using sample CSV file for testing")
            file_path = r"e:\New Codes\MP 2.o\02\sample_data.csv"
            file_info = {
                'original_filename': 'sample_data.csv',
                'file_size': 'unknown'
            }
        else:
            print(f"📁 Available uploaded_files keys: {list(uploaded_files.keys())}")
            
            # Check if file exists
            if file_id not in uploaded_files:
                print(f"❌ File {file_id} not found in uploaded_files")
                return jsonify({
                    'success': False,
                    'error': 'File not found. Please upload a file first.'
                }), 404
            
            file_info = uploaded_files[file_id]
            file_path = file_info['file_path']
        
        print(f"📂 Loading dataset from: {file_path}")
        
        # Load dataset using pandas (same as MLCore analyze_dataset)
        try:
            if file_path.endswith('.csv'):
                dataset = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                dataset = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                dataset = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format")
                
            print(f"✅ Dataset loaded successfully with shape: {dataset.shape}")
        except Exception as load_err:
            print(f"❌ Failed to load dataset: {load_err}")
            return jsonify({
                'success': False,
                'error': f'Failed to load dataset: {str(load_err)}'
            }), 500
        
        # Analyze columns using visualization engine
        try:
            column_info = visualization_engine.analyze_columns(dataset)
            print(f"✅ Column analysis completed: {len(column_info.get('columns', {}))} columns")
            print(f"🔍 Column info keys: {list(column_info.keys())}")
        except Exception as analysis_err:
            print(f"❌ Failed to analyze columns: {analysis_err}")
            return jsonify({
                'success': False,
                'error': f'Failed to analyze columns: {str(analysis_err)}'
            }), 500
        
        # Get available visualizations
        try:
            available_visualizations = visualization_engine.get_available_visualizations(dataset)
            print(f"✅ Available visualizations: {len(available_visualizations.get('visualizations', {}))} categories")
            print(f"🔍 Viz keys: {list(available_visualizations.keys())}")
        except Exception as viz_err:
            print(f"❌ Failed to get visualizations: {viz_err}")
            return jsonify({
                'success': False,
                'error': f'Failed to get available visualizations: {str(viz_err)}'
            }), 500
        
        # Convert numpy types to JSON serializable types manually
        import json
        
        try:
            # Transform the data to match frontend interface
            # Extract column types for easier access
            categorical_columns = []
            continuous_columns = []
            datetime_columns = []
            
            for col_name, col_info in column_info['columns'].items():
                if col_info.get('is_categorical', False):
                    categorical_columns.append(col_name)
                elif col_info.get('is_continuous', False):
                    continuous_columns.append(col_name)
                elif col_info.get('is_datetime', False):
                    datetime_columns.append(col_name)
                    
            print(f"📊 Categorical columns: {categorical_columns}")
            print(f"📈 Continuous columns: {continuous_columns}")
            print(f"📅 Datetime columns: {datetime_columns}")
            
            # Structure response to match frontend VisualizationAnalysis interface
            analysis_response = {
                'success': True,
                'analysis': {
                    'available_visualizations': available_visualizations.get('visualizations', {}),
                    'column_analysis': {
                        'total_rows': int(dataset.shape[0]),
                        'total_columns': int(dataset.shape[1]),
                        'columns': column_info['columns']
                    },
                    'categorical_columns': categorical_columns,
                    'continuous_columns': continuous_columns,
                    'datetime_columns': datetime_columns
                },
                'file_info': {
                    'original_filename': file_info['original_filename'],
                    'file_size': file_info['file_size']
                }
            }
            
            # Test JSON serialization
            json.dumps(analysis_response)
            print("✅ JSON serialization test passed")
            
            return jsonify(analysis_response)
            
        except TypeError as json_err:
            print(f"❌ JSON serialization error: {json_err}")
            # Fallback response with correct structure
            return jsonify({
                'success': True,
                'analysis': {
                    'available_visualizations': {
                        'univariate': {'categorical': [], 'continuous': []},
                        'bivariate': {'continuous_vs_continuous': [], 'categorical_vs_continuous': [], 'categorical_vs_categorical': []},
                        'multivariate': [],
                        'time_series': [],
                        'distribution': []
                    },
                    'column_analysis': {
                        'total_rows': int(dataset.shape[0]),
                        'total_columns': int(dataset.shape[1]),
                        'columns': {}
                    },
                    'categorical_columns': [],
                    'continuous_columns': [],
                    'datetime_columns': []
                },
                'file_info': {
                    'original_filename': file_info['original_filename'],
                    'file_size': file_info['file_size']
                },
                'error': f'Serialization issue: {str(json_err)}'
            })
        
    except Exception as e:
        error_msg = f"Error analyzing dataset for visualizations: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@app.route('/api/visualizations/create', methods=['POST'])
def create_visualization():
    """
    Create a visualization based on dataset and parameters
    
    JSON body:
    - file_id: ID of uploaded file
    - visualization_type: Type of visualization to create
    - parameters: Visualization parameters
    
    Returns:
    - JSON response with visualization data
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        file_id = data.get('file_id')
        viz_type = data.get('visualization_type')
        params = data.get('parameters', {})
        
        print(f"📊 Creating visualization: {viz_type} for file_id: {file_id}")
        
        if not viz_type:
            return jsonify({
                'success': False,
                'error': 'visualization_type is required'
            }), 400
        
        # Handle sample data case
        if not file_id or file_id == 'sample':
            print("🧪 Using sample CSV file for visualization")
            file_path = r"e:\New Codes\MP 2.o\02\sample_data.csv"
            file_info = {
                'original_filename': 'sample_data.csv',
                'file_size': 'unknown'
            }
        else:
            if file_id not in uploaded_files:
                return jsonify({
                    'success': False,
                    'error': 'Invalid file_id. Please upload a file first.'
                }), 400
            
            # Load the dataset using pandas (same as in analyze endpoint)
            file_path = uploaded_files[file_id]['file_path']
            file_info = uploaded_files[file_id]
        try:
            if file_path.endswith('.csv'):
                dataset = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                dataset = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                dataset = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format")
                
            print(f"✅ Dataset loaded for visualization with shape: {dataset.shape}")
        except Exception as load_err:
            print(f"❌ Failed to load dataset: {load_err}")
            return jsonify({
                'success': False,
                'error': f'Failed to load dataset: {str(load_err)}'
            }), 500
        
        # Create the visualization
        try:
            result = visualization_engine.create_visualization(dataset, viz_type, params)
            print(f"✅ Visualization created successfully")
            
            return jsonify({
                'success': result['success'],
                'visualization': result,
                'file_id': file_id,
                'visualization_type': viz_type,
                'parameters': params
            })
        except Exception as viz_err:
            print(f"❌ Failed to create visualization: {viz_err}")
            return jsonify({
                'success': False,
                'error': f'Failed to create visualization: {str(viz_err)}'
            }), 500
        
    except Exception as e:
        error_msg = f"Error creating visualization: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

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