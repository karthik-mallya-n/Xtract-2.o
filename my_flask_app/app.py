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

@app.route('/api/preview-columns', methods=['POST'])
def preview_columns():
    """
    Preview columns from uploaded file without saving
    
    Expected form data:
    - file: The dataset file
    
    Returns:
    - JSON response with column names
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
        
        # Read just the first few rows to get column names
        import pandas as pd
        import io
        
        file_content = file.read()
        file.seek(0)  # Reset file pointer
        
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), nrows=5)
            elif file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(io.BytesIO(file_content), nrows=5)
            elif file.filename.endswith('.json'):
                df = pd.read_json(io.StringIO(file_content.decode('utf-8')))
            else:
                return jsonify({
                    'success': False,
                    'error': 'Unsupported file format'
                }), 400
            
            columns = list(df.columns)
            
            return jsonify({
                'success': True,
                'columns': columns,
                'total_columns': len(columns)
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to read file: {str(e)}'
            }), 500
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Preview failed: {str(e)}'
        }), 500

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
        is_labeled = request.form.get('is_labeled', '').strip()
        data_type = request.form.get('data_type', '').strip()
        target_column = request.form.get('target_column', '').strip()
        
        # Validate required fields
        if not is_labeled:
            return jsonify({
                'success': False,
                'error': 'Missing questionnaire data. Please specify if data is labeled.'
            }), 400
        
        # For labeled data, data_type is required
        if is_labeled.lower() == 'labeled':
            if not data_type:
                return jsonify({
                    'success': False,
                    'error': 'Missing questionnaire data. Please specify data type (continuous or categorical) for labeled data.'
                }), 400
            # For labeled data, target_column is also required if available
            # (This will be validated later if needed)
        
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
                'data_type': data_type if is_labeled.lower() == 'labeled' else '',
                'target_column': target_column if is_labeled.lower() == 'labeled' else ''
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

        # Check if data is unlabeled to provide appropriate fallback
        is_unlabeled = user_answers.get('is_labeled', '').lower() == 'unlabeled'
        print(f"🔍 DEBUG: is_unlabeled = {is_unlabeled}, user_answers['is_labeled'] = {user_answers.get('is_labeled')}")
        
        # Define backend fallback recommendations based on data type
        if is_unlabeled:
            # Unlabeled data - recommend clustering models
            backend_fallback = {
                'recommended_models': [
                    {
                        'name': 'KMeans',
                        'description': 'Partition-based clustering algorithm that groups data into k clusters. Fast and efficient for large datasets.',
                        'accuracy_estimate': 85,
                        'reasoning': 'Best choice for unlabeled data when number of clusters is known or can be estimated.'
                    }
                ],
                'alternative_models': [
                    {
                        'name': 'DBSCAN',
                        'description': 'Density-based clustering that can find clusters of arbitrary shape and identify outliers.',
                        'accuracy_estimate': 80
                    },
                    {
                        'name': 'Gaussian Mixture Model (GMM)',
                        'description': 'Probabilistic clustering model that assumes data points are generated from a mixture of Gaussian distributions.',
                        'accuracy_estimate': 82
                    }
                ]
            }
        else:
            # Labeled data - recommend supervised models
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
        # Check if data is unlabeled for appropriate fallback
        # Note: user_answers should be available from file_info
        try:
            final_is_unlabeled = user_answers.get('is_labeled', '').lower() == 'unlabeled'
        except:
            final_is_unlabeled = False
        
        if final_is_unlabeled:
            fallback = {
                'recommended_models': [
                    {
                        'name': 'KMeans',
                        'description': 'Standard clustering fallback model.',
                        'accuracy_estimate': 80,
                        'reasoning': 'Safe default for unlabeled data'
                    }
                ],
                'alternative_models': []
            }
        else:
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
                return jsonify({
                    'success': False,
                    'error': f'File with ID {file_id} not found. Please upload a file first.'
                }), 404
        user_answers = file_info.get('user_answers', {})
        
        # Load the dataset
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
        
        # Determine if this is unlabeled data
        is_unlabeled = user_answers.get('is_labeled', '').lower() == 'unlabeled'
        
        # Check if this is a clustering model
        clustering_models = ['kmeans', 'dbscan', 'hierarchical clustering', 'gaussian mixture', 'gmm']
        is_clustering_model = any(cluster_model in model_name.lower() for cluster_model in clustering_models)
        
        # Determine target column based on data type
        target_column = None
        if is_unlabeled or is_clustering_model:
            # For unlabeled/clustering data, NO target column
            print(f"🔍 Unlabeled/clustering data detected - no target column needed")
            target_column = None
        else:
            # For labeled data, get target column from user answers
            if user_answers.get('is_labeled', '').lower() == 'labeled':
                if 'target_column' in user_answers and user_answers['target_column']:
                    target_column = user_answers['target_column']
                    if target_column not in df.columns:
                        return jsonify({
                            'success': False,
                            'error': f'Target column "{target_column}" not found in dataset. Available columns: {list(df.columns)}',
                            'available_columns': list(df.columns)
                        }), 400
                else:
                    # Return available columns for user selection
                    return jsonify({
                        'success': False,
                        'error': 'Target attribute not specified. Please select the target column.',
                        'available_columns': list(df.columns)
                    }), 400
            else:
                # Fallback: assume last column is target (shouldn't happen for labeled data)
                target_column = df.columns[-1]
                print(f"⚠️ Warning: Assuming last column as target: {target_column}")
        
        print(f"🚀 Starting training for model: {model_name}")
        print(f"📂 File path: {file_path}")
        print(f"📊 File ID: {file_id}")
        print(f"📄 Original filename: {file_info.get('original_filename', 'Unknown')}")
        print(f"🎯 Target column: {target_column if target_column else 'None (Unsupervised)'}")
        print(f"🔍 Is unlabeled: {is_unlabeled}, Is clustering model: {is_clustering_model}")
        
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
        is_unsupervised = result.get('performance', {}).get('model_type') == 'unsupervised' or is_unlabeled or is_clustering_model
        
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
                            'target_column': feature_data.get('target_column', target_column) if target_column else None,
                            'problem_type': result.get('performance', {}).get('model_type', 'unsupervised' if is_unsupervised else 'classification')
                        }
                        print(f"📊 Feature info loaded from artifacts: {len(feature_info.get('feature_names', []))} features")
                    else:
                        raise FileNotFoundError(f"Feature info file not found: {feature_info_path}")
            except Exception as e:
                print(f"⚠️ Could not load feature info from artifacts: {e}")
                # Try to get feature names from result first (these are the actual features used)
                if result.get('feature_names'):
                    feature_names = result['feature_names']
                    print(f"📊 Feature names from result: {feature_names}")
                elif result.get('training_details', {}).get('feature_names'):
                    feature_names = result['training_details']['feature_names']
                    print(f"📊 Feature names from training_details: {feature_names}")
                elif result.get('feature_info', {}).get('feature_names'):
                    feature_names = result['feature_info']['feature_names']
                    print(f"📊 Feature names from feature_info: {feature_names}")
                elif result.get('model_info', {}).get('feature_names'):
                    feature_names = result['model_info']['feature_names']
                    print(f"📊 Feature names from model_info: {feature_names}")
                else:
                    # Fallback: extract feature names from the dataset
                    try:
                        import pandas as pd
                        temp_df = pd.read_csv(file_path)
                        if is_unsupervised:
                            # For unsupervised, exclude ID columns (same logic as in training)
                            id_patterns = ['id', 'customerid', 'userid', 'user_id', 'customer_id', 'index', 'idx']
                            feature_names = []
                            for col in temp_df.columns:
                                col_lower = col.lower().strip()
                                if any(pattern in col_lower for pattern in id_patterns):
                                    unique_ratio = temp_df[col].nunique() / len(temp_df)
                                    if unique_ratio > 0.95:  # Exclude unique identifiers
                                        continue
                                feature_names.append(col)
                        else:
                            # For supervised, exclude target column
                            feature_names = [col for col in temp_df.columns if col != target_column] if target_column else list(temp_df.columns)
                        
                        print(f"📊 Feature names extracted from dataset: {feature_names}")
                    except Exception as inner_e:
                        print(f"❌ Could not extract features from dataset: {inner_e}")
                        feature_names = []
                
                # Get numeric and categorical columns from result or metadata
                numeric_cols = result.get('numeric_cols', result.get('training_details', {}).get('numeric_cols', []))
                categorical_cols = result.get('categorical_cols', result.get('training_details', {}).get('categorical_cols', []))
                
                feature_info = {
                    'feature_names': feature_names,
                    'target_column': target_column if not is_unsupervised else None,
                    'problem_type': 'unsupervised' if is_unsupervised else result.get('performance', {}).get('model_type', 'classification'),
                    'numeric_cols': numeric_cols,
                    'categorical_cols': categorical_cols
                }
                print(f"📊 Final feature info: {len(feature_names)} features - {feature_names}")
                print(f"📊 Numeric cols: {numeric_cols}")
                print(f"📊 Categorical cols: {categorical_cols}")
        else:
            # Training failed or no model_info, fallback to dataset
            # Try to get feature names from result first
            if result.get('feature_names'):
                feature_names = result['feature_names']
            elif result.get('training_details', {}).get('feature_names'):
                feature_names = result['training_details']['feature_names']
            elif result.get('feature_info', {}).get('feature_names'):
                feature_names = result['feature_info']['feature_names']
            else:
                try:
                    import pandas as pd
                    temp_df = pd.read_csv(file_path)
                    if is_unsupervised:
                        # For unsupervised, exclude ID columns (same logic as in training)
                        id_patterns = ['id', 'customerid', 'userid', 'user_id', 'customer_id', 'index', 'idx']
                        feature_names = []
                        for col in temp_df.columns:
                            col_lower = col.lower().strip()
                            if any(pattern in col_lower for pattern in id_patterns):
                                unique_ratio = temp_df[col].nunique() / len(temp_df)
                                if unique_ratio > 0.95:  # Exclude unique identifiers
                                    continue
                            feature_names.append(col)
                    else:
                        # For supervised, exclude target column
                        feature_names = [col for col in temp_df.columns if col != target_column] if target_column else list(temp_df.columns)
                except Exception as e:
                    print(f"❌ Could not extract features from dataset: {e}")
                    feature_names = []
            
            # Get numeric and categorical columns from result if available
            numeric_cols = result.get('numeric_cols', result.get('training_details', {}).get('numeric_cols', []))
            categorical_cols = result.get('categorical_cols', result.get('training_details', {}).get('categorical_cols', []))
            
            feature_info = {
                'feature_names': feature_names,
                'target_column': target_column if not is_unsupervised else None,
                'problem_type': 'unsupervised' if is_unsupervised else 'classification',
                'numeric_cols': numeric_cols,
                'categorical_cols': categorical_cols
            }
            print(f"📊 Feature info (fallback): {len(feature_names)} features - {feature_names}")
            print(f"📊 Numeric cols (fallback): {numeric_cols}")
            print(f"📊 Categorical cols (fallback): {categorical_cols}")
        
        # Format response to match frontend expectations
        is_unsupervised_result = result.get('performance', {}).get('model_type') == 'unsupervised' or is_unlabeled or is_clustering_model
        
        if result.get('success'):
            # Get original training_details from result if it exists
            original_training_details = result.get('training_details', {})
            
            # Get feature names from multiple possible sources
            feature_names_from_result = (
                result.get('feature_names') or
                result.get('training_details', {}).get('feature_names') or
                result.get('feature_info', {}).get('feature_names') or
                result.get('model_info', {}).get('feature_names') or
                feature_info.get('feature_names')
            )
            
            # Format the result to match TrainingResponse interface
            formatted_result = {
                'success': True,
                'model_name': result.get('model_info', {}).get('name', model_name),
                'model_folder': result.get('model_info', {}).get('model_directory', ''),
                'main_score': result.get('performance', {}).get('main_score', result.get('performance', {}).get('silhouette_score', 0.5)),
                'score_name': result.get('performance', {}).get('score_name', 'accuracy' if not is_unsupervised_result else 'silhouette_score'),
                'problem_type': result.get('performance', {}).get('model_type', 'unsupervised' if is_unsupervised_result else 'classification'),
                'threshold_met': result.get('performance', {}).get('silhouette_score', 0) > 0.3 if is_unsupervised_result else result.get('threshold_met', False),
                'performance': result.get('performance', {}),
                'model_info': result.get('model_info', {}),
                'training_details': {
                    # Preserve original training_details (includes feature_names)
                    **original_training_details,
                    # Override with performance data if not present
                    'training_samples': original_training_details.get('training_samples', result.get('performance', {}).get('n_samples', 0)),
                    'features': original_training_details.get('features', result.get('performance', {}).get('n_features', 0)),
                    'training_time': original_training_details.get('training_time', result.get('performance', {}).get('training_time', 0)),
                    # Ensure feature_names are included
                    'feature_names': feature_names_from_result or original_training_details.get('feature_names', [])
                }
            }
            
            # Update feature_info with names from result if available
            if feature_names_from_result:
                feature_info['feature_names'] = feature_names_from_result
            
            # Ensure feature_info includes numeric and categorical columns
            if 'numeric_cols' not in feature_info:
                feature_info['numeric_cols'] = result.get('numeric_cols', result.get('training_details', {}).get('numeric_cols', []))
            if 'categorical_cols' not in feature_info:
                feature_info['categorical_cols'] = result.get('categorical_cols', result.get('training_details', {}).get('categorical_cols', []))
            
            print(f"📊 Formatted result training_details.feature_names: {formatted_result['training_details'].get('feature_names', 'NOT FOUND')}")
            print(f"📊 Feature info feature_names: {feature_info.get('feature_names', 'NOT FOUND')}")
            print(f"📊 Result feature_names: {result.get('feature_names', 'NOT FOUND')}")
            print(f"📊 Result training_details.feature_names: {result.get('training_details', {}).get('feature_names', 'NOT FOUND')}")
            
            # Add clustering-specific info if unsupervised
            if is_unsupervised_result:
                formatted_result['performance']['n_clusters'] = result.get('performance', {}).get('n_clusters')
                formatted_result['performance']['cluster_distribution'] = result.get('performance', {}).get('cluster_distribution', {})
        else:
            formatted_result = result
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'model_name': model_name,
            'result': formatted_result,
            'feature_info': feature_info,
            'message': 'Training completed successfully!' + (' (Unsupervised)' if is_unsupervised_result else '')
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
        data = request.get_json()
        
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
        
        # Get all folders with their timestamps and sort by most recent
        folder_timestamps = []
        for folder in all_model_folders:
            ts = get_folder_timestamp(folder)
            if ts != "00000000_000000":
                folder_timestamps.append((folder, ts))
        
        if not folder_timestamps:
            return jsonify({
                'success': False,
                'error': 'No valid trained models found. Please train a model first.'
            }), 404
        
        # Sort by timestamp (most recent first)
        folder_timestamps.sort(key=lambda x: x[1], reverse=True)
        model_folder = folder_timestamps[0][0]
        
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
        
        # Get feature names from metadata (may be nested in performance or at top level)
        feature_names = (
            metadata.get('feature_names') or
            metadata.get('performance', {}).get('feature_names') or
            []
        )
        
        # For unsupervised models, try to get feature names from the model directory
        if not feature_names and metadata.get('model_type') == 'unsupervised':
            # Try to load from feature_info file if it exists
            feature_info_files = glob.glob(os.path.join(model_folder, 'feature_info_*.json'))
            if feature_info_files:
                feature_info_files.sort(reverse=True)
                try:
                    with open(feature_info_files[0], 'r') as f:
                        feature_info = json.load(f)
                        feature_names = feature_info.get('feature_names', [])
                        print(f"📊 Loaded feature names from feature_info: {feature_names}")
                except Exception as e:
                    print(f"⚠️ Could not load feature_info: {e}")
        
        target_column = metadata.get('target_column')
        problem_type = metadata.get('model_type', metadata.get('problem_type', 'regression'))
        
        # For unsupervised models, target_column is None, so use all features
        if problem_type == 'unsupervised' or target_column is None:
            features_for_prediction = feature_names
        else:
            # For supervised models, exclude target column
            features_for_prediction = [f for f in feature_names if f != target_column]
        
        print(f"📊 Feature names from metadata: {feature_names}")
        print(f"📊 Features for prediction: {features_for_prediction}")
        print(f"📊 Target column: {target_column}")
        print(f"📊 Problem type: {problem_type}")
        
        # Now map features array to proper feature names if needed
        if 'features' in data and input_data is None:
            features_array = data.get('features', [])
            
            if len(features_array) != len(features_for_prediction):
                print(f"⚠️  Feature count mismatch: received {len(features_array)}, expected {len(features_for_prediction)}")
                print(f"⚠️  Expected features: {features_for_prediction}")
                return jsonify({
                    'success': False,
                    'error': f'Feature count mismatch. Expected {len(features_for_prediction)} features, got {len(features_array)}. Expected features: {features_for_prediction}'
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
        # First, ensure we have all required features in the correct order
        if problem_type == 'unsupervised' or target_column is None:
            features_for_prediction = feature_names
        else:
            features_for_prediction = [f for f in feature_names if f != target_column]
        
        # Build input_data dict with all features in correct order
        ordered_input_data = {}
        for feature in features_for_prediction:
            if feature in input_data:
                ordered_input_data[feature] = input_data[feature]
            else:
                print(f"⚠️  Missing feature '{feature}', setting to 0")
                ordered_input_data[feature] = 0
        
        # Create DataFrame with features in the correct order
        input_df = pd.DataFrame([ordered_input_data])
        
        # Get categorical and numeric columns from metadata
        categorical_cols_meta = metadata.get('categorical_cols', [])
        numeric_cols_meta = metadata.get('numeric_cols', [])
        
        # FIRST: Ensure categorical columns are treated as strings (for proper encoding)
        # This must happen BEFORE numeric conversion to prevent overwriting
        if problem_type == 'unsupervised':
            for col in categorical_cols_meta:
                if col in input_df.columns:
                    # Keep as string - don't convert to numeric
                    input_df[col] = input_df[col].astype(str)
                    print(f"   ✓ Converted {col} to string for categorical encoding (value: '{input_df[col].iloc[0]}')")
        
        # THEN: Ensure numeric columns are numeric (only for non-categorical columns)
        for col in numeric_cols_meta:
            if col in input_df.columns and col not in categorical_cols_meta:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                # Fill NaN with 0 if conversion failed
                if input_df[col].isna().any():
                    input_df[col] = input_df[col].fillna(0)
                    print(f"   ⚠️  Converted {col} to numeric (filled NaN with 0)")
        
        print(f"🔄 Input data shape: {input_df.shape}")
        print(f"🔄 Input features: {list(input_df.columns)}")
        print(f"🔄 Input data values: {input_df.iloc[0].to_dict()}")
        print(f"🔄 Input data types: {input_df.dtypes.to_dict()}")
        
        # Use direct model loading (bypass advanced_model_trainer for compatibility)
        try:
            # Load the pipeline model directly - check for multiple possible filenames
            # Exclude preprocessing files (scaler, imputer, encoders, etc.)
            exclude_patterns = ['scaler', 'imputer', 'encoder', 'feature_info', 'metadata', 'transformed_data']
            
            # Pattern 1: model_*.joblib (supervised models)
            model_files = [f for f in glob.glob(os.path.join(model_folder, 'model_*.joblib')) 
                          if not any(pattern in os.path.basename(f).lower() for pattern in exclude_patterns)]
            
            # Pattern 2: *model*.joblib (unsupervised models, e.g., kmeans_*.joblib, gaussian_mixture_model_(gmm)_*.joblib)
            if not model_files:
                all_files = glob.glob(os.path.join(model_folder, '*model*.joblib'))
                model_files = [f for f in all_files 
                              if not any(pattern in os.path.basename(f).lower() for pattern in exclude_patterns)]
            
            # Pattern 3: Any .joblib file that's not a preprocessing artifact
            if not model_files:
                all_files = glob.glob(os.path.join(model_folder, '*.joblib'))
                model_files = [f for f in all_files 
                              if not any(pattern in os.path.basename(f).lower() for pattern in exclude_patterns)]
            
            print(f"🔍 Found model files (excluding preprocessing): {model_files}")
            
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
                # Use the most recent model file (by modification time)
                model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                model_path = model_files[0]
                print(f"📁 Selected model file: {os.path.basename(model_path)}")
            
            print(f"📁 Loading model from: {model_path}")
            
            # Load the trained model
            import joblib
            model = joblib.load(model_path)
            
            print(f"✅ Model loaded successfully: {type(model)}")
            
            # For unsupervised models, we need to apply preprocessing using saved transformers
            if problem_type == 'unsupervised':
                print("🔧 Applying preprocessing for unsupervised model using saved transformers...")
                
                # Load preprocessing transformers from metadata
                preprocessing_artifacts = metadata.get('preprocessing_artifacts', {})
                numeric_cols = metadata.get('numeric_cols', [])
                categorical_cols = metadata.get('categorical_cols', [])
                
                # Create a copy for preprocessing
                X_pred = input_df.copy()
                
                print(f"🔧 BEFORE PREPROCESSING:")
                print(f"   Input values: {X_pred.iloc[0].to_dict()}")
                
                # Load and apply numeric imputer if available
                if preprocessing_artifacts.get('numeric_imputer'):
                    imputer_path = os.path.join(model_folder, preprocessing_artifacts['numeric_imputer'])
                    if os.path.exists(imputer_path):
                        numeric_imputer = joblib.load(imputer_path)
                        if numeric_cols:
                            # Only impute numeric columns that exist
                            numeric_cols_to_impute = [col for col in numeric_cols if col in X_pred.columns]
                            if numeric_cols_to_impute:
                                # Use .loc to ensure proper assignment
                                imputed_values = numeric_imputer.transform(X_pred[numeric_cols_to_impute])
                                for idx, col in enumerate(numeric_cols_to_impute):
                                    X_pred.loc[:, col] = imputed_values[:, idx]
                                print(f"   ✓ Applied numeric imputation to {numeric_cols_to_impute}")
                                print(f"   After imputation: {X_pred[numeric_cols_to_impute].iloc[0].to_dict()}")
                
                # Load and apply label encoders for categorical columns
                if preprocessing_artifacts.get('label_encoders'):
                    encoders_path = os.path.join(model_folder, preprocessing_artifacts['label_encoders'])
                    if os.path.exists(encoders_path):
                        label_encoders = joblib.load(encoders_path)
                        for col in categorical_cols:
                            if col in X_pred.columns and col in label_encoders:
                                le = label_encoders[col]
                                # Handle unseen values by using the first class
                                try:
                                    original_value = str(X_pred[col].iloc[0])
                                    encoded_values = le.transform(X_pred[col].astype(str))
                                    X_pred.loc[:, col] = encoded_values
                                    print(f"   ✓ Encoded {col}: '{original_value}' -> {encoded_values[0]}")
                                except ValueError as e:
                                    # Unseen value, use first class as default
                                    default_value = le.transform([le.classes_[0]])[0]
                                    X_pred.loc[:, col] = default_value
                                    print(f"   ⚠️  Unseen value in {col}, using default: {default_value}")
                
                # Load and apply scaler for numeric columns
                if preprocessing_artifacts.get('scaler') and numeric_cols:
                    scaler_path = os.path.join(model_folder, preprocessing_artifacts['scaler'])
                    if os.path.exists(scaler_path):
                        scaler = joblib.load(scaler_path)
                        # Only scale numeric columns that exist
                        numeric_cols_to_scale = [col for col in numeric_cols if col in X_pred.columns]
                        if numeric_cols_to_scale:
                            print(f"   Before scaling: {X_pred[numeric_cols_to_scale].iloc[0].to_dict()}")
                            # Use .loc to ensure proper assignment
                            scaled_values = scaler.transform(X_pred[numeric_cols_to_scale])
                            for idx, col in enumerate(numeric_cols_to_scale):
                                X_pred.loc[:, col] = scaled_values[:, idx]
                            print(f"   ✓ Scaled {len(numeric_cols_to_scale)} numeric columns using saved scaler")
                            print(f"   After scaling: {X_pred[numeric_cols_to_scale].iloc[0].to_dict()}")
                
                # Ensure columns are in the correct order (matching training)
                X_pred = X_pred[feature_names]  # Reorder to match training order
                
                # Convert to numpy array for prediction
                X_array = X_pred.values if hasattr(X_pred, 'values') else X_pred
                print(f"📊 Preprocessed input shape: {X_array.shape}")
                print(f"📊 Preprocessed input array: {X_array[0] if len(X_array) > 0 else 'N/A'}")
                print(f"📊 Preprocessed feature values: {dict(zip(feature_names, X_array[0])) if len(X_array) > 0 else 'N/A'}")
                
                # Make prediction (cluster assignment for unsupervised)
                print("🔮 Making cluster prediction...")
                predictions = model.predict(X_array)
                cluster_id = int(predictions[0]) if len(predictions) > 0 else 0
                
                # Get cluster distribution for user-friendly output
                performance_data = metadata.get('performance', {})
                cluster_distribution = performance_data.get('cluster_distribution', {})
                cluster_size = cluster_distribution.get(str(cluster_id), 0)
                total_samples = performance_data.get('n_samples', metadata.get('n_samples', 0))
                cluster_percentage = (cluster_size / total_samples * 100) if total_samples > 0 else 0
                
                # Format user-friendly prediction
                prediction = f"Cluster {cluster_id}"
                cluster_prediction = {
                    'cluster_id': cluster_id,
                    'cluster_label': f"Cluster {cluster_id}",
                    'cluster_size': cluster_size,
                    'cluster_percentage': round(cluster_percentage, 2),
                    'total_samples': total_samples
                }
                
                print(f"✅ Cluster assignment: {prediction} ({cluster_size} samples, {cluster_percentage:.1f}%)")
            else:
                # For supervised models, check if it's a pipeline
                if hasattr(model, 'predict') and hasattr(model, 'steps'):
                    # It's a pipeline
                    print(f"📊 Pipeline steps: {getattr(model, 'steps', 'N/A')}")
                    predictions = model.predict(input_df)
                elif hasattr(model, 'predict'):
                    # It's a model, apply preprocessing if needed
                    predictions = model.predict(input_df.values)
                else:
                    raise ValueError("Model does not have predict method")
                
                prediction = predictions[0] if len(predictions) > 0 else 0
            
            print(f"✅ Prediction result: {prediction}")
            print(f"✅ Prediction type: {type(prediction)}")
            
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
        
        # Format prediction for user-friendly display
        if problem_type == 'unsupervised' and 'cluster_prediction' in locals():
            # For clustering, use the cluster prediction details
            prediction_display = f"{cluster_prediction.get('cluster_label', f'Cluster {prediction}')} ({cluster_prediction.get('cluster_percentage', 0):.1f}% of data)"
            prediction_decoded = prediction_display
        else:
            # For supervised models, use the prediction as-is
            prediction_decoded = str(prediction)
            cluster_prediction = None
        
        return jsonify({
            'success': True,
            'prediction': prediction_decoded,
            'raw_prediction': float(prediction) if isinstance(prediction, (int, float, np.number)) and problem_type != 'unsupervised' else str(prediction),
            'cluster_info': cluster_prediction if problem_type == 'unsupervised' else None,  # Include cluster details for unsupervised
            'confidence': None,  # Confidence not available in new system yet
            'probabilities': None,  # Probabilities not available in new system yet
            'model_info': {
                'model_name': metadata.get('model_name', 'Unknown'),
                'training_date': metadata.get('timestamp', ''),
                'accuracy': metadata.get('test_score', metadata.get('performance', {}).get('accuracy', metadata.get('performance', {}).get('r2_score', 0))),
                'problem_type': problem_type
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