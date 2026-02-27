/**
 * API Client for ML Platform
 * Handles all communication between Next.js frontend and Flask backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://backend:5000';

export interface UploadResponse {
  success: boolean;
  file_id?: string;
  filename?: string;
  file_size?: number;
  message?: string;
  error?: string;
  available_columns?: string[];
}

export interface ModelRecommendation {
  success: boolean;
  file_id?: string;
  dataset_info?: {
    total_rows: number;
    total_columns: number;
    numeric_columns: number;
    categorical_columns: number;
  };
  user_answers?: {
    is_labeled: string;
    data_type: string;
  };
  recommendations?: {
    recommended_models?: Array<{
      name: string;
      description: string;
      accuracy_estimate: number;
      reasoning: string;
    }>;
    alternative_models?: Array<{
      name: string;
      description: string;
      accuracy_estimate: number;
    }>;
  };
  raw_llm_response?: string;
  error?: string;
}

export interface TrainingResponse {
  success: boolean;
  task_id?: string;
  file_id?: string;
  model_name?: string;
  message?: string;
  error?: string;
  feature_info?: {
    feature_names: string[];
    target_column: string;
    problem_type: string;
  };
  result?: {
    success: boolean;
    model_folder: string;
    model_name: string;
    main_score: number;
    score_name: string;
    problem_type: string;
    threshold_met: boolean;
    performance: {
      accuracy: number;
      cv_accuracy: number;
      cv_std: number;
      classification_report: Record<string, {
        precision: number;
        recall: number;
        'f1-score': number;
        support: number;
      }>;
    };
    best_params: Record<string, string | number | boolean | null>;
  };
}

export interface TrainingStatus {
  success: boolean;
  task_id?: string;
  state?: string;
  progress?: number;
  status?: string;
  current_step?: string;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  is_complete?: boolean;
  results?: {
    model_name: string;
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    training_time: number;
  };
  error?: string;
}

export interface PredictionResponse {
  success: boolean;
  prediction?: string;
  raw_prediction?: number | string;
  confidence?: number;
  probabilities?: {
    probabilities: number[];
    confidence: number;
  };
  model_info?: {
    model_name: string;
    training_date: string;
    accuracy: number;
  };
  input_data?: Record<string, string | number>;
  error?: string;
}

export interface ModelInfo {
  filename: string;
  model_name: string;
  training_date: string;
  accuracy: number;
  dataset_info: {
    total_rows?: number;
    total_columns?: number;
    features?: string[];
    target?: string;
  };
}

export interface ModelsListResponse {
  success: boolean;
  models?: ModelInfo[];
  count?: number;
  error?: string;
}

class APIClient {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  /**
   * Generic fetch wrapper with error handling
   */
  private async fetch<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });
      const contentType = response.headers.get('content-type') || '';
      let data: unknown = null;
      if (contentType.includes('application/json')) {
        try {
          data = await response.json();
        } catch {
          data = null;
        }
      } else {
        // Fallback to text for non-JSON responses
        try {
          data = await response.text();
        } catch {
          data = null;
        }
      }

      if (!response.ok) {
        // Surface backend error message if present
        const message = (data && typeof data === 'object' && 'error' in data && typeof data.error === 'string') 
          ? data.error 
          : (data && typeof data === 'object' && 'message' in data && typeof data.message === 'string')
          ? data.message
          : `HTTP error ${response.status}`;
        const error = new Error(message) as Error & { status: number; payload: unknown };
        error.status = response.status;
        error.payload = data;
        throw error;
      }

      return data as T;
    } catch (error) {
      console.error(`API Error [${endpoint}]:`, error);
      throw error;
    }
  }

  /**
   * Check API health
   */
  async healthCheck(): Promise<{ status: string; timestamp: string; version: string }> {
    return this.fetch('/api/health');
  }

  /**
   * Upload a dataset file
   */
  async uploadFile(
    file: File,
    isLabeled: string,
    dataType: string
  ): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('is_labeled', isLabeled);
    formData.append('data_type', dataType);

    const url = `${this.baseURL}/api/upload`;

    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      return await response.json();
    } catch (error) {
      console.error('Upload Error:', error);
      throw error;
    }
  }

  /**
   * Get model recommendations for uploaded file
   */
  async getModelRecommendations(fileId: string): Promise<ModelRecommendation> {
    return this.fetch(`/api/recommend-model?file_id=${fileId}`);
  }

  /**
   * Start model training
   */
  async startTraining(fileId: string, modelName: string): Promise<TrainingResponse> {
    return this.fetch('/api/train', {
      method: 'POST',
      body: JSON.stringify({
        file_id: fileId,
        model_name: modelName,
      }),
    });
  }

  /**
   * Get training status
   */
  async getTrainingStatus(taskId: string): Promise<TrainingStatus> {
    return this.fetch(`/api/training-status/${taskId}`);
  }

  /**
   * Make prediction with trained model
   */
  async makePrediction(
    fileId: string,
    inputData: Record<string, string | number>
  ): Promise<PredictionResponse> {
    return this.fetch('/api/predict', {
      method: 'POST',
      body: JSON.stringify({
        file_id: fileId,
        input_data: inputData,
      }),
    });
  }

  /**
   * List all available trained models
   */
  async listModels(): Promise<ModelsListResponse> {
    return this.fetch('/api/models');
  }
}

// Create singleton instance
export const apiClient = new APIClient();

// Utility functions for common API operations
export const api = {
  /**
   * Check if backend is available
   */
  isBackendAvailable: async (): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(3000) // 3 second timeout
      });
      return response.ok;
    } catch {
      return false;
    }
  },

  /**
   * Upload file with progress tracking
   */
  uploadWithProgress: async (
    file: File,
    isLabeled: string,
    dataType: string,
    targetColumn?: string,
    onProgress?: (progress: number) => void,
    selectedTrainingColumns?: string[]
  ): Promise<UploadResponse> => {
    return new Promise((resolve, reject) => {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('is_labeled', isLabeled || '');
      
      // Only send data_type for labeled data
      if (isLabeled === 'labeled') {
        formData.append('data_type', dataType || '');
      }
      // For unlabeled data, don't send data_type at all
      
      if (targetColumn) {
        formData.append('target_column', targetColumn);
      }

      // Include selected training columns
      if (selectedTrainingColumns && selectedTrainingColumns.length > 0) {
        formData.append('selected_columns', JSON.stringify(selectedTrainingColumns));
      }

      const xhr = new XMLHttpRequest();

      // Track upload progress
      if (onProgress) {
        xhr.upload.addEventListener('progress', (event) => {
          if (event.lengthComputable) {
            const progress = Math.round((event.loaded / event.total) * 100);
            onProgress(progress);
          }
        });
      }

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const response = JSON.parse(xhr.responseText);
            resolve(response);
          } catch {
            reject(new Error('Failed to parse response'));
          }
        } else {
          // Try to parse error message from response
          let errorMessage = `HTTP error! status: ${xhr.status}`;
          try {
            const errorResponse = JSON.parse(xhr.responseText);
            if (errorResponse.error) {
              errorMessage = errorResponse.error;
            } else if (errorResponse.message) {
              errorMessage = errorResponse.message;
            }
          } catch {
            // If parsing fails, use default message
          }
          reject(new Error(errorMessage));
        }
      });

      xhr.addEventListener('error', () => {
        reject(new Error('Network error'));
      });

      xhr.open('POST', `${API_BASE_URL}/api/upload`);
      xhr.send(formData);
    });
  }
};