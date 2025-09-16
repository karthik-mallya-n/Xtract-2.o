'use client';

import { useState, useEffect } from 'react';
import { CheckCircle, Brain, Target, Loader2, AlertCircle } from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { apiClient, ModelRecommendation } from '@/lib/api';

/**
 * Model Card Component - Displays individual ML model information
 */
interface ModelCardProps {
  name: string;
  description: string;
  accuracy: number;
  isRecommended?: boolean;
  onSelect: () => void;
  isSelected: boolean;
}

function ModelCard({ name, description, accuracy, isRecommended, onSelect, isSelected }: ModelCardProps) {
  return (
    <div 
      className={`bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow duration-200 p-6 cursor-pointer border-2 ${
        isSelected ? 'border-blue-500' : 'border-transparent'
      } ${isRecommended ? 'ring-2 ring-green-500' : ''}`}
      onClick={onSelect}
    >
      {isRecommended && (
        <div className="flex items-center mb-3">
          <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
          <span className="text-sm font-medium text-green-700 bg-green-100 px-2 py-1 rounded-full">
            AI Recommended
          </span>
        </div>
      )}
      
      <div className="flex items-center mb-3">
        <Brain className="h-6 w-6 text-blue-600 mr-3" />
        <h3 className="text-xl font-semibold text-gray-900">{name}</h3>
      </div>
      
      <p className="text-gray-600 mb-4">{description}</p>
      
      <div className="flex justify-between items-center">
        <div className="text-sm text-gray-500">
          Expected Accuracy: <span className="font-semibold text-gray-900">{accuracy}%</span>
        </div>
        
        <button
          className={`px-4 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
            isSelected
              ? 'bg-blue-600 text-white'
              : 'bg-blue-50 text-blue-600 hover:bg-blue-100'
          }`}
        >
          {isSelected ? 'Selected' : 'Select'}
        </button>
      </div>
    </div>
  );
}

/**
 * Model Selection Page - Displays AI recommendations and alternative models
 */
export default function SelectModelPage() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [recommendations, setRecommendations] = useState<ModelRecommendation | null>(null);
  const [error, setError] = useState<string>('');
  const [fileId, setFileId] = useState<string>('');

  // Load file ID from localStorage and fetch recommendations
  useEffect(() => {
    const loadRecommendations = async () => {
      const storedFileId = localStorage.getItem('currentFileId');
      
      if (!storedFileId) {
        setError('No file uploaded. Please upload a dataset first.');
        setIsLoading(false);
        return;
      }

      setFileId(storedFileId);

      try {
        const response = await apiClient.getModelRecommendations(storedFileId);
        
        if (response.success) {
          setRecommendations(response);
        } else {
          setError(response.error || 'Failed to get model recommendations');
        }
      } catch (err) {
        console.error('Error fetching recommendations:', err);
        setError('Failed to connect to the backend. Please ensure the Flask server is running.');
      } finally {
        setIsLoading(false);
      }
    };

    loadRecommendations();
  }, []);

  const handleModelSelect = (modelId: string) => {
    setSelectedModel(modelId);
  };

  const handleStartTraining = () => {
    if (!selectedModel || !fileId) {
      setError('Please select a model first');
      return;
    }

    // Store selected model and navigate to training page
    localStorage.setItem('selectedModel', selectedModel);
    router.push('/training-status');
  };

  // Mock data for fallback when LLM is not available
  const fallbackModels = [
    {
      id: 'random-forest',
      name: 'Random Forest',
      description: 'Ensemble learning method that operates by constructing multiple decision trees. Great for both classification and regression tasks.',
      accuracy: 87,
      isRecommended: true,
    },
    {
      id: 'svm',
      name: 'Support Vector Machine',
      description: 'Powerful algorithm for classification and regression. Works well with high-dimensional data and complex patterns.',
      accuracy: 84,
      isRecommended: false,
    },
    {
      id: 'gradient-boosting',
      name: 'Gradient Boosting',
      description: 'Sequential ensemble method that builds models iteratively. Excellent for achieving high accuracy.',
      accuracy: 89,
      isRecommended: false,
    },
    {
      id: 'logistic-regression',
      name: 'Logistic Regression',
      description: 'Statistical method for binary classification. Simple, interpretable, and effective for linearly separable data.',
      accuracy: 81,
      isRecommended: false,
    },
  ];

  // Combine API recommendations with fallback models
  const getModelsToDisplay = () => {
    if (!recommendations || !recommendations.recommendations) {
      return fallbackModels;
    }

    const apiModels = [
      ...(recommendations.recommendations.recommended_models || []).map((model, index) => ({
        id: model.name.toLowerCase().replace(/\s+/g, '-'),
        name: model.name,
        description: model.description,
        accuracy: model.accuracy_estimate,
        isRecommended: index === 0, // First recommended model is primary
      })),
      ...(recommendations.recommendations.alternative_models || []).map(model => ({
        id: model.name.toLowerCase().replace(/\s+/g, '-'),
        name: model.name,
        description: model.description,
        accuracy: model.accuracy_estimate,
        isRecommended: false,
      })),
    ];

    return apiModels.length > 0 ? apiModels : fallbackModels;
  };

  const models = getModelsToDisplay();
  const recommendedModel = models.find(model => model.isRecommended);
  const otherModels = models.filter(model => !model.isRecommended);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-16 w-16 text-blue-600 animate-spin mx-auto mb-6" />
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Analyzing Your Data
          </h2>
          <p className="text-lg text-gray-600 max-w-md">
            Our AI is examining your dataset to recommend the best machine learning models for your specific use case.
          </p>
          <div className="mt-6 bg-white rounded-lg p-4 shadow-md max-w-md mx-auto">
            <div className="text-sm text-gray-500 space-y-2">
              <div>✓ Data structure analysis complete</div>
              <div>✓ Feature correlation calculated</div>
              <div className="text-blue-600">→ Evaluating model compatibility...</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-8 text-center">
          <AlertCircle className="h-16 w-16 text-red-500 mx-auto mb-6" />
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Error Loading Recommendations
          </h2>
          <p className="text-gray-600 mb-6">{error}</p>
          <div className="space-y-3">
            <Link
              href="/upload"
              className="block w-full px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-md transition-colors duration-200"
            >
              Upload New Dataset
            </Link>
            <button
              onClick={() => window.location.reload()}
              className="block w-full px-4 py-2 text-sm font-medium text-blue-600 bg-blue-50 hover:bg-blue-100 rounded-md transition-colors duration-200"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-4">
            <Target className="h-12 w-12 text-blue-600" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Model Recommendations
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Based on your dataset analysis, we&apos;ve identified the best machine learning models for your project.
          </p>
          
          {/* Dataset Info */}
          {recommendations?.dataset_info && (
            <div className="mt-6 bg-white rounded-lg p-4 shadow-md max-w-2xl mx-auto">
              <h3 className="font-semibold text-gray-900 mb-2">Dataset Overview</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Rows:</span>
                  <span className="ml-1 font-medium">{recommendations.dataset_info.total_rows}</span>
                </div>
                <div>
                  <span className="text-gray-500">Columns:</span>
                  <span className="ml-1 font-medium">{recommendations.dataset_info.total_columns}</span>
                </div>
                <div>
                  <span className="text-gray-500">Numeric:</span>
                  <span className="ml-1 font-medium">{recommendations.dataset_info.numeric_columns}</span>
                </div>
                <div>
                  <span className="text-gray-500">Categorical:</span>
                  <span className="ml-1 font-medium">{recommendations.dataset_info.categorical_columns}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Recommended Model Section */}
        {recommendedModel && (
          <div className="mb-12">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                Recommended Model
              </h2>
              <p className="text-gray-600">
                Our AI suggests this model based on your data characteristics
              </p>
            </div>
            
            <div className="max-w-2xl mx-auto">
              <ModelCard
                name={recommendedModel.name}
                description={recommendedModel.description}
                accuracy={recommendedModel.accuracy}
                isRecommended={true}
                onSelect={() => handleModelSelect(recommendedModel.id)}
                isSelected={selectedModel === recommendedModel.id}
              />
            </div>
          </div>
        )}

        {/* Alternative Models Section */}
        {otherModels.length > 0 && (
          <div className="mb-12">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                Alternative Models
              </h2>
              <p className="text-gray-600">
                Other models that could work well with your dataset
              </p>
            </div>
            
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {otherModels.map((model) => (
                <ModelCard
                  key={model.id}
                  name={model.name}
                  description={model.description}
                  accuracy={model.accuracy}
                  isRecommended={false}
                  onSelect={() => handleModelSelect(model.id)}
                  isSelected={selectedModel === model.id}
                />
              ))}
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex justify-center space-x-4">
          <Link
            href="/upload"
            className="px-6 py-3 text-base font-medium text-gray-700 bg-white border border-gray-300 hover:bg-gray-50 rounded-lg transition-colors duration-200"
          >
            Upload Different Dataset
          </Link>
          
          <button
            onClick={handleStartTraining}
            disabled={!selectedModel}
            className={`px-6 py-3 text-base font-medium rounded-lg transition-colors duration-200 ${
              selectedModel
                ? 'text-white bg-blue-600 hover:bg-blue-700'
                : 'text-gray-400 bg-gray-200 cursor-not-allowed'
            }`}
          >
            Start Training
          </button>
        </div>
      </div>
    </div>
  );
}
                🎯 Our Top Recommendation
              </h2>
              <p className="text-gray-600">
                This model is optimally suited for your dataset characteristics
              </p>
            </div>
            
            <div className="max-w-2xl mx-auto">
              <ModelCard
                name={recommendedModel.name}
                description={recommendedModel.description}
                accuracy={recommendedModel.accuracy}
                isRecommended={true}
                onSelect={() => handleModelSelect(recommendedModel.id)}
                isSelected={selectedModel === recommendedModel.id}
              />
            </div>
          </div>
        )}

        {/* Other Models Section */}
        <div className="mb-12">
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              Alternative Models
            </h2>
            <p className="text-gray-600">
              Explore other models that could work well with your data
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {otherModels.map((model) => (
              <ModelCard
                key={model.id}
                name={model.name}
                description={model.description}
                accuracy={model.accuracy}
                isRecommended={false}
                onSelect={() => handleModelSelect(model.id)}
                isSelected={selectedModel === model.id}
              />
            ))}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="text-center">
          <div className="bg-white rounded-lg shadow-md p-6 max-w-md mx-auto">
            {selectedModel ? (
              <div>
                <p className="text-gray-600 mb-4">
                  Ready to train your {models.find(m => m.id === selectedModel)?.name} model?
                </p>
                <Link
                  href="/training-status"
                  className="inline-flex items-center px-8 py-3 text-lg font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors duration-200"
                >
                  Start Training
                </Link>
              </div>
            ) : (
              <div>
                <p className="text-gray-500 mb-4">
                  Please select a model to continue
                </p>
                <button
                  disabled
                  className="px-8 py-3 text-lg font-medium text-gray-400 bg-gray-200 rounded-lg cursor-not-allowed"
                >
                  Select a Model First
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}