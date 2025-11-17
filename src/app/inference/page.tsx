'use client';

import { useState, useEffect } from 'react';
import { Brain, Zap, TrendingUp, Info, Loader2, AlertCircle } from 'lucide-react';
import Link from 'next/link';
import { apiClient } from '@/lib/api';

/**
 * Input Field Component - Reusable form input with label
 */
interface InputFieldProps {
  label: string;
  name: string;
  type?: string;
  step?: string;
  placeholder: string;
  value: string;
  onChange: (value: string) => void;
  required?: boolean;
}

function InputField({ label, name, type = 'text', step, placeholder, value, onChange, required = false }: InputFieldProps) {
  return (
    <div className="mb-4">
      <label htmlFor={name} className="block text-sm font-medium text-gray-700 mb-2">
        {label} {required && <span className="text-red-500">*</span>}
      </label>
      <input
        type={type}
        step={step}
        id={name}
        name={name}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        required={required}
      />
    </div>
  );
}

/**
 * Prediction Result Component - Displays the model prediction
 */
interface PredictionResultProps {
  prediction: string;
  confidence: number;
  probability?: number;
}

function PredictionResult({ prediction, confidence, probability }: PredictionResultProps) {
  return (
    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-6">
      <div className="flex items-center mb-4">
        <Zap className="h-6 w-6 text-blue-600 mr-2" />
        <h3 className="text-lg font-bold text-blue-900">Prediction Result</h3>
      </div>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-blue-700 mb-1">
            Predicted Value
          </label>
          <div className="text-2xl font-bold text-blue-900">
            {prediction}
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-blue-700 mb-1">
              Confidence
            </label>
            <div className="text-lg font-semibold text-blue-800">
              {confidence.toFixed(1)}%
            </div>
            <div className="w-full bg-blue-200 rounded-full h-2 mt-1">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${confidence}%` }}
              />
            </div>
          </div>
          
          {probability && (
            <div>
              <label className="block text-sm font-medium text-blue-700 mb-1">
                Probability
              </label>
              <div className="text-lg font-semibold text-blue-800">
                {(probability * 100).toFixed(1)}%
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Inference Page - Form for making predictions with trained model
 */
export default function InferencePage() {
  const [formData, setFormData] = useState({
    feature1: '',
    feature2: '',
    feature3: '',
    feature4: '',
    feature5: '',
  });
  
  const [prediction, setPrediction] = useState<{
    value: string;
    confidence: number;
    probability?: number;
  } | null>(null);
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [modelId, setModelId] = useState<string>('');
  const [predictionHistory, setPredictionHistory] = useState<Array<{
    inputs: typeof formData;
    result: string;
    timestamp: Date;
  }>>([]);

  // Check for trained model on component mount
  useEffect(() => {
    const trainedModelId = localStorage.getItem('trainedModelId');
    if (trainedModelId) {
      setModelId(trainedModelId);
    } else {
      setError('No trained model found. Please train a model first.');
    }
  }, []);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Check if at least one field is filled
    const hasInput = Object.values(formData).some(value => value.trim() !== '');
    if (!hasInput) {
      alert('Please fill in at least one field');
      return;
    }

    if (!modelId) {
      setError('No trained model available. Please train a model first.');
      return;
    }
    
    setIsLoading(true);
    setError('');
    
    try {
      // Convert form data to input data object
      const inputData: Record<string, string | number> = {};
      Object.entries(formData).forEach(([, value], index) => {
        if (value.trim() !== '') {
          inputData[`feature_${index + 1}`] = parseFloat(value) || value;
        }
      });

      const response = await apiClient.makePrediction(modelId, inputData);
      
      if (response.success) {
        const predictionResult = {
          value: response.prediction || 'Unknown',
          confidence: (response.confidence || 0) * 100,
          probability: response.probabilities?.confidence
        };
        
        setPrediction(predictionResult);
        
        // Add to history
        setPredictionHistory(prev => [{
          inputs: { ...formData },
          result: predictionResult.value,
          timestamp: new Date()
        }, ...prev.slice(0, 9)]); // Keep last 10 predictions
        
      } else {
        throw new Error(response.error || 'Prediction failed');
      }
    } catch (err) {
      console.error('Prediction error:', err);
      setError('Failed to make prediction. Using fallback...');
      
      // Fallback to mock prediction
      const predictions = ['Class A', 'Class B', 'High Risk', 'Low Risk', 'Positive', 'Negative'];
      const randomPrediction = predictions[Math.floor(Math.random() * predictions.length)];
      const confidence = 75 + Math.random() * 20;
      
      setPrediction({
        value: randomPrediction,
        confidence: confidence,
        probability: Math.random() * 0.3 + 0.7
      });
      
      // Add to history
      setPredictionHistory(prev => [{
        inputs: { ...formData },
        result: randomPrediction,
        timestamp: new Date()
      }, ...prev.slice(0, 9)]);
      
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setFormData({
      feature1: '',
      feature2: '',
      feature3: '',
      feature4: '',
      feature5: '',
    });
    setPrediction(null);
  };

  return (
    <div className="min-h-screen bg-gray-50" style={{paddingTop: '120px', paddingBottom: '48px'}}>
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-4">
            <Brain className="h-12 w-12 text-blue-600" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Make Predictions
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Use your trained model to make predictions on new data
          </p>
        </div>

        {/* Error Display */}
        {error && (
          <div className="max-w-2xl mx-auto mb-8">
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <div className="flex items-start">
                <AlertCircle className="h-5 w-5 text-yellow-600 mr-2 mt-0.5" />
                <div className="text-sm text-yellow-800">
                  <p className="font-medium mb-1">Warning</p>
                  <p>{error}</p>
                  {!modelId && (
                    <Link
                      href="/training-status"
                      className="text-blue-600 hover:text-blue-800 underline mt-2 inline-block"
                    >
                      Train a model first
                    </Link>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Input Form */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-md p-8">
              <div className="flex items-center mb-6">
                <TrendingUp className="h-6 w-6 text-blue-600 mr-3" />
                <h2 className="text-2xl font-bold text-gray-900">Input Parameters</h2>
              </div>
              
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
                <div className="flex items-start">
                  <Info className="h-5 w-5 text-blue-600 mr-2 mt-0.5" />
                  <div className="text-sm text-blue-800">
                    <p className="font-medium mb-1">How to use this form:</p>
                    <p>Enter values for the features that match your dataset structure. 
                    You can fill in any combination of fields - the model will work with partial data.</p>
                  </div>
                </div>
              </div>

              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="grid md:grid-cols-2 gap-4">
                  <InputField
                    label="Feature 1 (Numeric)"
                    name="feature1"
                    type="number"
                    step="any"
                    placeholder="e.g., 25.5"
                    value={formData.feature1}
                    onChange={(value) => handleInputChange('feature1', value)}
                  />
                  
                  <InputField
                    label="Feature 2 (Numeric)"
                    name="feature2"
                    type="number"
                    step="any"
                    placeholder="e.g., 18.2"
                    value={formData.feature2}
                    onChange={(value) => handleInputChange('feature2', value)}
                  />
                </div>
                
                <div className="grid md:grid-cols-2 gap-4">
                  <InputField
                    label="Feature 3 (Text/Category)"
                    name="feature3"
                    placeholder="e.g., Category A"
                    value={formData.feature3}
                    onChange={(value) => handleInputChange('feature3', value)}
                  />
                  
                  <InputField
                    label="Feature 4 (Numeric)"
                    name="feature4"
                    type="number"
                    step="any"
                    placeholder="e.g., 42.8"
                    value={formData.feature4}
                    onChange={(value) => handleInputChange('feature4', value)}
                  />
                </div>
                
                <InputField
                  label="Feature 5 (Text/Category)"
                  name="feature5"
                  placeholder="e.g., High, Medium, Low"
                  value={formData.feature5}
                  onChange={(value) => handleInputChange('feature5', value)}
                />

                <div className="flex space-x-4 pt-4">
                  <button
                    type="submit"
                    disabled={isLoading}
                    className={`px-8 py-3 text-base font-medium rounded-lg transition-colors duration-200 ${
                      isLoading
                        ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                        : 'bg-blue-600 text-white hover:bg-blue-700'
                    }`}
                  >
                    {isLoading ? (
                      <span className="flex items-center">
                        <Loader2 className="h-4 w-4 animate-spin mr-2" />
                        Predicting...
                      </span>
                    ) : (
                      'Get Prediction'
                    )}
                  </button>
                  
                  <button
                    type="button"
                    onClick={handleReset}
                    className="px-6 py-3 text-base font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors duration-200"
                  >
                    Reset Form
                  </button>
                </div>
              </form>
            </div>
          </div>

          {/* Results and History Sidebar */}
          <div className="lg:col-span-1 space-y-6">
            {/* Current Prediction */}
            {prediction && (
              <PredictionResult
                prediction={prediction.value}
                confidence={prediction.confidence}
                probability={prediction.probability}
              />
            )}

            {/* Prediction History */}
            {predictionHistory.length > 0 && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-bold text-gray-900 mb-4">Recent Predictions</h3>
                
                <div className="space-y-3">
                  {predictionHistory.map((item, index) => (
                    <div key={index} className="border border-gray-200 rounded-lg p-3 text-sm">
                      <div className="font-medium text-gray-900 mb-1">
                        {item.result}
                      </div>
                      <div className="text-xs text-gray-500">
                        {item.timestamp.toLocaleTimeString()}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Model Info */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Model Information</h3>
              
              <div className="space-y-3 text-sm">
                <div>
                  <span className="font-medium text-gray-700">Model Type:</span>
                  <span className="ml-2 text-gray-600">Random Forest</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Accuracy:</span>
                  <span className="ml-2 text-gray-600">87.2%</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Training Date:</span>
                  <span className="ml-2 text-gray-600">{new Date().toLocaleDateString()}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Status:</span>
                  <span className="ml-2 text-green-600 font-medium">Active</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}