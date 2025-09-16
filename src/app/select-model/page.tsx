'use client';

import { useState, useEffect } from 'react';
import { CheckCircle, Brain, Target, Loader2 } from 'lucide-react';
import Link from 'next/link';

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
  const [isLoading, setIsLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [recommendationReady, setRecommendationReady] = useState(false);

  // Mock data for available models
  const models = [
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
      id: 'neural-network',
      name: 'Neural Network',
      description: 'Deep learning model inspired by biological neural networks. Excellent for complex pattern recognition tasks.',
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

  useEffect(() => {
    // Simulate AI recommendation process
    const timer = setTimeout(() => {
      setIsLoading(false);
      setRecommendationReady(true);
    }, 3000);

    return () => clearTimeout(timer);
  }, []);

  const handleModelSelect = (modelId: string) => {
    setSelectedModel(modelId);
  };

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
            Based on your dataset analysis, we've identified the best machine learning models for your project.
          </p>
        </div>

        {/* Recommended Model Section */}
        {recommendedModel && (
          <div className="mb-12">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
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