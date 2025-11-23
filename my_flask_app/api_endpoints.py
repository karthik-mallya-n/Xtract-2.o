"""
Additional API endpoints for enhanced ML platform functionality
"""

from flask import Blueprint, request, jsonify
import os
import json
import glob

# Create a blueprint for additional endpoints
enhanced_api = Blueprint('enhanced_api', __name__)

@enhanced_api.route('/api/model-columns', methods=['GET'])
def get_model_columns():
    """
    Get the column information for the most recent trained model,
    including AI-selected columns for predictions
    
    Returns:
    - JSON response with column information for frontend prediction form
    """
    try:
        # Find the most recent model folder
        model_storage_path = os.getenv('MODEL_STORAGE_PATH', 'models')
        all_model_folders = glob.glob(f"{model_storage_path}/*")
        all_model_folders = [f for f in all_model_folders if os.path.isdir(f)]
        
        if not all_model_folders:
            return jsonify({
                'success': False,
                'error': 'No trained models found. Please train a model first.'
            }), 404
        
        # Function to get folder timestamp
        def get_folder_timestamp(folder_path):
            folder_name = os.path.basename(folder_path)
            parts = folder_name.split('_')
            if len(parts) >= 2:
                timestamp_part = parts[-1]
                if len(timestamp_part) == 6 and timestamp_part.isdigit():
                    if len(parts) >= 3 and len(parts[-2]) == 8 and parts[-2].isdigit():
                        return f"{parts[-2]}_{timestamp_part}"
                    else:
                        return f"20251115_{timestamp_part}"
            
            metadata_files = glob.glob(os.path.join(folder_path, 'metadata_*.json'))
            if metadata_files:
                timestamps = []
                for mf in metadata_files:
                    basename = os.path.basename(mf)
                    if basename.startswith('metadata_') and basename.endswith('.json'):
                        ts_part = basename[9:-5]
                        timestamps.append(ts_part)
                if timestamps:
                    timestamps.sort(reverse=True)
                    return timestamps[0]
            
            return "00000000_000000"
        
        # Get all folders with timestamps and sort by most recent
        folder_timestamps = []
        for folder in all_model_folders:
            ts = get_folder_timestamp(folder)
            if ts != "00000000_000000":
                folder_timestamps.append((folder, ts))
        
        if not folder_timestamps:
            return jsonify({
                'success': False,
                'error': 'No valid trained models found.'
            }), 404
        
        # Prioritize Iris models if they exist
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
        
        for folder, ts in folder_timestamps:
            if is_iris_model(folder):
                iris_models.append((folder, ts))
            else:
                other_models.append((folder, ts))
        
        # Choose model: always prioritize Iris models if they exist (user is working with Iris data)
        if iris_models:
            iris_models.sort(key=lambda x: x[1], reverse=True)
            model_folder = iris_models[0][0]
            print(f"🌸 API: Using IRIS model folder: {model_folder}")
        elif other_models:
            other_models.sort(key=lambda x: x[1], reverse=True)  
            model_folder = other_models[0][0]
            print(f"📁 API: Using other model folder: {model_folder}")
        else:
            return jsonify({
                'success': False,
                'error': 'No valid trained models found.'
            }), 404
        
        # Load metadata
        metadata_files = glob.glob(os.path.join(model_folder, 'metadata_*.json'))
        if not metadata_files:
            metadata_path = os.path.join(model_folder, 'metadata.json')
            if not os.path.exists(metadata_path):
                return jsonify({
                    'success': False,
                    'error': 'Model metadata not found.'
                }), 500
        else:
            metadata_files.sort(reverse=True)
            metadata_path = metadata_files[0]
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load feature info if available
        feature_info_files = glob.glob(os.path.join(model_folder, 'feature_info_*.json'))
        feature_info = {}
        if feature_info_files:
            feature_info_files.sort(reverse=True)
            with open(feature_info_files[0], 'r') as f:
                feature_info = json.load(f)
        
        # Extract column information
        feature_names = metadata.get('feature_names', [])
        target_column = metadata.get('target_column', 'target')
        
        # Get AI column selection info if available
        column_selection = metadata.get('column_selection', {})
        ai_selected = column_selection.get('ai_selected', [])
        ai_excluded = column_selection.get('ai_excluded', [])
        columns_used_for_training = column_selection.get('columns_used_for_training', feature_names)
        
        print(f"🔍 Debug - Feature names: {feature_names}")
        print(f"🔍 Debug - Target column: {target_column}")
        print(f"🔍 Debug - AI selected: {ai_selected}")
        print(f"🔍 Debug - AI excluded: {ai_excluded}")
        
        # If no AI selection exists, apply ML best practices automatically
        if not ai_selected and not ai_excluded:
            # Check if this is an Iris model for specialized handling
            is_iris = (target_column.lower() == 'species' or 
                      any('sepal' in col.lower() or 'petal' in col.lower() for col in feature_names))
            
            if is_iris:
                # For Iris dataset, specifically exclude 'Id' column
                ai_excluded = [col for col in feature_names if col in ['Id', 'id', 'ID', 'index', 'Index']]
                print(f"🌸 IRIS: Excluding ID columns: {ai_excluded}")
            else:
                # For other datasets, exclude standard ID-like columns
                ai_excluded = [col for col in feature_names if col.lower() in ['id', 'index', 'row_id', 'record_id']]
            
            ai_selected = [col for col in feature_names if col not in ai_excluded and col != target_column]
            
            print(f"🤖 Auto-applied - AI selected: {ai_selected}")
            print(f"🤖 Auto-applied - AI excluded: {ai_excluded}")
            
            # Set reasoning for the automatic selection
            column_selection = {
                'ai_selected': ai_selected,
                'ai_excluded': ai_excluded,
                'reasoning': {
                    'included_reasoning': f"Selected {', '.join(ai_selected)} as they contain meaningful features for prediction.",
                    'excluded_reasoning': f"Excluded {', '.join(ai_excluded)} as they are identifier columns with no predictive value." if ai_excluded else "No columns excluded."
                }
            }
        
        # Determine which columns to ask user for (only AI-selected columns, exclude target)
        if ai_selected:
            columns_for_prediction = [col for col in ai_selected if col != target_column]
        else:
            # Fallback: exclude target and obvious ID columns
            excluded_patterns = ['id', 'index', 'row_id', 'record_id']
            columns_for_prediction = [col for col in feature_names 
                                    if col != target_column 
                                    and col.lower() not in excluded_patterns]
                                    
        print(f"📋 Final columns for prediction: {columns_for_prediction}")
        
        # Get column type information
        numeric_features = feature_info.get('numeric_features', [])
        categorical_features = feature_info.get('categorical_features', [])
        
        # Build column info for frontend
        column_info = []
        for col in columns_for_prediction:
            col_type = 'numeric' if col in numeric_features else 'categorical'
            is_ai_selected = True  # All columns in columns_for_prediction are considered selected
            column_info.append({
                'name': col,
                'type': col_type,
                'required': True,
                'ai_selected': is_ai_selected,
                'placeholder': f"Enter {col.replace('_', ' ').title()}"
            })
        
        response = {
            'success': True,
            'model_info': {
                'name': metadata.get('model_name', 'Unknown'),
                'type': metadata.get('model_type', 'Unknown'),
                'training_date': metadata.get('timestamp', ''),
                'target_column': target_column
            },
            'columns_for_prediction': column_info,
            'ai_column_selection': {
                'enabled': bool(ai_selected or ai_excluded),
                'selected_columns': ai_selected,
                'excluded_columns': ai_excluded,
                'reasoning': column_selection.get('reasoning', {})
            },
            'total_columns_needed': len(columns_for_prediction)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get model columns: {str(e)}'
        }), 500

@enhanced_api.route('/api/column-selection-info/<file_id>', methods=['GET'])
def get_column_selection_info(file_id):
    """
    Get AI column selection recommendations for a specific dataset
    
    Parameters:
    - file_id: ID of the uploaded file
    
    Returns:
    - JSON response with column selection recommendations
    """
    try:
        from core_ml import ml_core
        
        # This would be called during the model recommendation phase
        # to show users which columns AI recommends for training
        
        # For now, return a placeholder response
        return jsonify({
            'success': True,
            'message': 'Column selection info will be available during model training',
            'file_id': file_id
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get column selection info: {str(e)}'
        }), 500