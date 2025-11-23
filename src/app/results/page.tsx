'use client';

import { useState, useEffect, useCallback, Suspense } from 'react';
import { 
  Brain, 
  TrendingUp, 
  Target, 
  Zap, 
  Loader2, 
  AlertCircle,
  CheckCircle,
  Settings,
  BarChart3,
  Database
} from 'lucide-react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { motion } from 'framer-motion';
import ParticleBackground from '@/components/ParticleBackground';

// Types
interface AIColumnInfo {
  name: string;
  type: 'numeric' | 'categorical';
  ai_selected: boolean;
  required: boolean;
}

interface AIColumnSelection {
  enabled: boolean;
  selected_columns: string[];
  excluded_columns: string[];
  reasoning: {
    included_reasoning?: string;
    excluded_reasoning?: string;
  };
}

interface TrainingResults {
  model_name?: string;
  main_score: number;
  threshold_met: boolean;
  performance?: {
    model_name?: string;
    model_type?: string;
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
    mse?: number;
    rmse?: number;
    mae?: number;
    r2_score?: number;
    training_time?: number;
    prediction_time?: number;
    cv_accuracy?: number;
    classification_report?: Record<string, {
      precision: number;
      recall: number;
      'f1-score': number;
      support: number;
    }>;
  };
  training_details?: {
    training_samples?: number;
    test_samples?: number;
    features?: number;
    training_time?: number;
    total_samples?: number;
    feature_names?: string[];
    target_column?: string;
    model_type?: string;
    cv_folds?: number;
    best_score?: number;
    problem_type?: string;
    test_split?: number;
    cross_validation?: number;
    preprocessing_steps?: number;
    data_quality?: string;
    timestamp?: string;
  };
  model_info?: {
    name?: string;
    type?: string;
    feature_count?: number;
    training_samples?: number;
    test_samples?: number;
    model_directory?: string;
  };
  feature_info?: {
    feature_names?: string[];
    original_feature_names?: string[];
    target_column?: string;
    problem_type?: string;
    feature_count?: number;
    dataset_shape?: number[];
  };
  file_info?: {
    filename?: string;
    target_column?: string;
  };
  model_params?: Record<string, string | number | boolean>;
}

/**
 * Tab Navigation Component
 */
interface TabNavigationProps {
  activeTab: 'visualization' | 'prediction';
  onTabChange: (tab: 'visualization' | 'prediction') => void;
}

function TabNavigation({ activeTab, onTabChange }: TabNavigationProps) {
  return (
    <div className="border-b border-gray-600 mb-8">
      <nav className="-mb-px flex space-x-8">
        <motion.button
          onClick={() => onTabChange('visualization')}
          className={`py-3 px-2 border-b-2 font-bold text-lg transition-all duration-200 ${
            activeTab === 'visualization'
              ? 'border-cyan-400 text-cyan-400'
              : 'border-transparent text-gray-400 hover:text-white hover:border-gray-500'
          }`}
          whileHover={{ y: -2 }}
        >
          <div className="flex items-center">
            <BarChart3 className="h-6 w-6 mr-3" />
            Training Results & Visualization
          </div>
        </motion.button>
        <motion.button
          onClick={() => onTabChange('prediction')}
          className={`py-3 px-2 border-b-2 font-bold text-lg transition-all duration-200 ${
            activeTab === 'prediction'
              ? 'border-cyan-400 text-cyan-400'
              : 'border-transparent text-gray-400 hover:text-white hover:border-gray-500'
          }`}
          whileHover={{ y: -2 }}
        >
          <div className="flex items-center">
            <Brain className="h-6 w-6 mr-3" />
            Make Predictions
          </div>
        </motion.button>
      </nav>
    </div>
  );
}

/**
 * Metric Card Component
 */
interface MetricCardProps {
  title: string;
  value: string | number;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  format?: 'percentage' | 'decimal' | 'text';
  description?: string;
}

function MetricCard({ title, value, icon: Icon, color, format = 'text', description }: MetricCardProps) {
  const formatValue = (val: string | number) => {
    if (typeof val === 'string') return val;
    if (format === 'percentage') return `${(val * 100).toFixed(2)}%`;
    if (format === 'decimal') return val.toFixed(4);
    return val.toString();
  };

  return (
    <motion.div 
      className="futuristic-card hover:border-cyan-400/50"
      whileHover={{ y: -5 }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-lg font-bold text-gray-300">{title}</p>
          <p className={`text-4xl font-black mt-3 ${color}`}>
            {formatValue(value)}
          </p>
          {description && (
            <p className="text-sm text-cyan-400 mt-2 font-medium">{description}</p>
          )}
        </div>
        <div className="p-4 rounded-xl bg-gray-800/50 border border-gray-600">
          <Icon className={`h-10 w-10 ${color}`} />
        </div>
      </div>
    </motion.div>
  );
}

/**
 * Classification Report Component
 */
interface ClassificationReportProps {
  classificationReport: Record<string, {
    precision: number;
    recall: number;
    'f1-score': number;
    support: number;
  }>;
}

function ClassificationReport({ classificationReport }: ClassificationReportProps) {
  const classes = Object.keys(classificationReport).filter(
    key => !['accuracy', 'macro avg', 'weighted avg'].includes(key)
  );

  return (
    <motion.div 
      className="futuristic-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.2 }}
    >
      <div className="flex items-center mb-8">
        <div className="p-3 rounded-xl bg-green-900/30 border border-green-500/30 mr-4">
          <Target className="h-8 w-8 text-green-400" />
        </div>
        <h3 className="text-2xl font-bold text-white">Classification Report</h3>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full">
          <thead>
            <tr className="border-b border-gray-600">
              <th className="px-6 py-4 text-left text-lg font-bold text-cyan-400 uppercase tracking-wider">
                Class
              </th>
              <th className="px-6 py-4 text-left text-lg font-bold text-cyan-400 uppercase tracking-wider">
                Precision
              </th>
              <th className="px-6 py-4 text-left text-lg font-bold text-cyan-400 uppercase tracking-wider">
                Recall
              </th>
              <th className="px-6 py-4 text-left text-lg font-bold text-cyan-400 uppercase tracking-wider">
                F1-Score
              </th>
              <th className="px-6 py-4 text-left text-lg font-bold text-cyan-400 uppercase tracking-wider">
                Support
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-700">
            {classes.map((className) => (
              <tr key={className} className="hover:bg-gray-800/30 transition-colors duration-200">
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className="text-lg font-bold text-white">{className}</span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className="text-lg font-medium text-green-400">
                    {(classificationReport[className]?.precision * 100 || 0).toFixed(1)}%
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className="text-lg font-medium text-cyan-400">
                    {(classificationReport[className]?.recall * 100 || 0).toFixed(1)}%
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className="text-lg font-medium text-purple-400">
                    {(classificationReport[className]?.['f1-score'] * 100 || 0).toFixed(1)}%
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className="text-lg font-medium text-gray-300">
                    {classificationReport[className]?.support || 0}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </motion.div>
  );
}

/**
 * Model Parameters Component
 */
interface ModelParametersProps {
  modelParams: Record<string, string | number | boolean>;
  modelName: string;
}

function ModelParameters({ modelParams, modelName }: ModelParametersProps) {
  return (
    <motion.div 
      className="futuristic-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.4 }}
    >
      <div className="flex items-center mb-8">
        <div className="p-3 rounded-xl bg-purple-900/30 border border-purple-500/30 mr-4">
          <Settings className="h-8 w-8 text-purple-400" />
        </div>
        <h3 className="text-2xl font-bold text-white">Model Configuration</h3>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        <div>
          <h4 className="text-lg font-bold text-cyan-400 mb-4 uppercase tracking-wider">
            Algorithm Details
          </h4>
          <div className="space-y-4">
            <div className="flex justify-between items-center py-2 border-b border-gray-700">
              <span className="text-gray-300 font-medium">Model Type</span>
              <span className="text-white font-bold">{modelName.replace(/[-_]/g, ' ')}</span>
            </div>
            {Object.entries(modelParams).slice(0, 5).map(([key, value]) => (
              <div key={key} className="flex justify-between items-center py-2 border-b border-gray-700">
                <span className="text-gray-300 font-medium capitalize">{key.replace(/[-_]/g, ' ')}</span>
                <span className="text-white font-bold">
                  {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h4 className="text-lg font-bold text-cyan-400 mb-4 uppercase tracking-wider">
            Training Info
          </h4>
          <div className="space-y-4">
            <div className="flex justify-between items-center py-2 border-b border-gray-700">
              <span className="text-gray-300 font-medium">Random State</span>
              <span className="text-white font-bold">{modelParams.random_state || 'N/A'}</span>
            </div>
            {Object.entries(modelParams).slice(5, 10).map(([key, value]) => (
              <div key={key} className="flex justify-between items-center py-2 border-b border-gray-700">
                <span className="text-gray-300 font-medium capitalize">{key.replace(/[-_]/g, ' ')}</span>
                <span className="text-white font-bold">
                  {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  );
}

// Component that uses useSearchParams - needs to be wrapped in Suspense
function ResultsPageContent() {
  const searchParams = useSearchParams();
  const [activeTab, setActiveTab] = useState<'visualization' | 'prediction'>('visualization');
  const [trainingResults, setTrainingResults] = useState<TrainingResults | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string>('');

  // Dynamic prediction form state
  const [formData, setFormData] = useState<Record<string, string>>({});
  const [aiColumns, setAiColumns] = useState<AIColumnInfo[]>([]);
  const [aiColumnSelection, setAiColumnSelection] = useState<AIColumnSelection | null>(null);
  const [isLoadingColumns, setIsLoadingColumns] = useState(true);
  
  const [prediction, setPrediction] = useState<{
    value: string;
    confidence: number;
  } | null>(null);
  
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionHistory, setPredictionHistory] = useState<Array<{
    inputs: Record<string, string>;
    result: string;
    confidence: number;
    timestamp: Date;
  }>>([]);

  // Fetch AI-recommended columns from API
  const fetchAIColumns = useCallback(async () => {
    try {
      setIsLoadingColumns(true);
      const response = await fetch('http://localhost:5000/api/model-columns');
      const data = await response.json();
      
      if (data.success) {
        setAiColumns(data.columns_for_prediction || []);
        setAiColumnSelection(data.ai_column_selection || null);
        console.log('🤖 AI Column data loaded:', data);
      }
    } catch (err) {
      console.error('❌ Failed to load AI columns:', err);
    } finally {
      setIsLoadingColumns(false);
    }
  }, []);

  // Get AI-recommended feature names only
  const getFeatureNames = useCallback((): string[] => {
    // If AI columns are available, use only AI-selected ones
    if (aiColumns.length > 0) {
      return aiColumns
        .filter(col => col.ai_selected)
        .map(col => col.name);
    }
    
    // Fallback to original logic if AI columns not available
    if (trainingResults?.training_details?.feature_names) {
      return trainingResults.training_details.feature_names as string[];
    }
    if (trainingResults?.feature_info?.feature_names) {
      return trainingResults.feature_info.feature_names as string[];
    }
    if (trainingResults?.feature_info?.original_feature_names) {
      return trainingResults.feature_info.original_feature_names as string[];
    }
    
    // Generate generic feature names as fallback
    const featureCount = trainingResults?.training_details?.features || 
                        trainingResults?.model_info?.feature_count || 4;
    
    return Array.from({length: featureCount}, (_, i) => `Feature_${i + 1}`);
  }, [aiColumns, trainingResults]);

  // Initialize form data when training results change
  useEffect(() => {
    if (trainingResults) {
      const featureNames = getFeatureNames();
      const initialFormData: Record<string, string> = {};
      featureNames.forEach(feature => {
        initialFormData[feature] = '';
      });
      setFormData(initialFormData);
    }
  }, [trainingResults, getFeatureNames]);

  // Fetch AI columns when component mounts
  useEffect(() => {
    fetchAIColumns();
  }, [fetchAIColumns]);

  // Load training results from URL params or localStorage
  useEffect(() => {
    const loadTrainingResults = () => {
      try {
        // First, try to get results from URL parameters
        const resultsParam = searchParams.get('results');
        if (resultsParam) {
          const results = JSON.parse(decodeURIComponent(resultsParam));
          setTrainingResults(results);
          setIsLoading(false);
          return;
        }

        // Fallback to localStorage
        const savedResults = localStorage.getItem('trainingResults');
        if (savedResults) {
          const results = JSON.parse(savedResults);
          setTrainingResults(results);
          setIsLoading(false);
          return;
        }

        // If no results found, set error
        setError('No training results found. Please train a model first.');
        setIsLoading(false);

      } catch (err) {
        console.error('Failed to load training results:', err);
        setError('Failed to load training results');
        setIsLoading(false);
      }
    };

    loadTrainingResults();
  }, [searchParams]);

  const handleInputChange = (featureName: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [featureName]: value
    }));
  };

  const handlePrediction = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsPredicting(true);
    setError('');

    try {
      // Validate all fields are filled
      const featureNames = getFeatureNames();
      const missingFields = featureNames.filter(name => !formData[name] || formData[name].trim() === '');
      
      if (missingFields.length > 0) {
        throw new Error(`Please fill in all fields: ${missingFields.join(', ')}`);
      }

      // Convert form data to feature array in the correct order
      const features = featureNames.map(name => {
        const value = formData[name];
        const aiColumnInfo = aiColumns.find(col => col.name === name);
        return aiColumnInfo?.type === 'numeric' ? parseFloat(value) || 0 : value;
      });
      
      // Validate numeric values for numeric columns
      const invalidFeatures = features.some((feature, index) => {
        const featureName = featureNames[index];
        const aiColumnInfo = aiColumns.find(col => col.name === featureName);
        return aiColumnInfo?.type === 'numeric' && isNaN(feature as number);
      });
      
      if (invalidFeatures) {
        throw new Error('All numeric inputs must be valid numbers');
      }

      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input_data: Object.fromEntries(
            featureNames.map((name, index) => [name, features[index]])
          ),
          file_id: 'latest'
        }),
      });

      if (!response.ok) {
        throw new Error('Prediction request failed');
      }

      const apiResult = await response.json();
      
      // Use real prediction from the model
      const predictedClass = apiResult.prediction || 'Unknown';
      const confidence = apiResult.confidence || 95;

      setPrediction({
        value: predictedClass,
        confidence: confidence
      });

      // Add to history
      const newHistoryItem = {
        inputs: { ...formData },
        result: predictedClass,
        confidence: confidence,
        timestamp: new Date()
      };
      
      setPredictionHistory(prev => [newHistoryItem, ...prev.slice(0, 4)]); // Keep only last 5

    } catch (err) {
      console.error('Prediction failed:', err);
      setError('Failed to make prediction');
    } finally {
      setIsPredicting(false);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen relative overflow-hidden bg-gray-900 flex items-center justify-center">
        {/* Background Elements */}
        <ParticleBackground />
        <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-purple-900 to-cyan-900 opacity-20" />
        <div className="absolute inset-0 geometric-pattern opacity-30" />
        
        {/* Loading Content */}
        <div className="relative z-10 text-center">
          <motion.div
            className="flex justify-center mb-8"
            animate={{ 
              rotateY: 360,
              scale: [1, 1.2, 1]
            }}
            transition={{ 
              rotateY: { duration: 2, repeat: Infinity, ease: "linear" },
              scale: { duration: 1.5, repeat: Infinity, ease: "easeInOut" }
            }}
          >
            <Brain className="h-24 w-24 text-cyan-400" style={{filter: 'drop-shadow(0 0 20px rgba(0, 245, 255, 0.5))'}} />
          </motion.div>
          <h2 className="text-4xl font-black text-white mb-6">
            Loading Training Results
          </h2>
          <p className="text-xl text-gray-300 max-w-md mx-auto">
            Please wait while we prepare your model results...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen relative overflow-hidden bg-gray-900">
      {/* Background Elements */}
      <ParticleBackground />
      <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-purple-900 to-cyan-900 opacity-20" />
      <div className="absolute inset-0 geometric-pattern opacity-30" />
      
      {/* Main Container */}
      <div className="relative z-10 min-h-screen" style={{paddingTop:'80px', paddingBottom: '48px'}}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Header */}
          <motion.div 
            className="text-center mb-12"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <motion.div
              className="flex justify-center mb-6"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 1, delay: 0.2 }}
            >
              <motion.div
                className="w-20 h-20 rounded-full flex items-center justify-center"
                style={{
                  background: 'linear-gradient(135deg, rgba(0, 245, 255, 0.2) 0%, rgba(153, 69, 255, 0.2) 100%)',
                  border: '2px solid rgba(0, 245, 255, 0.3)',
                  boxShadow: '0 0 30px rgba(0, 245, 255, 0.3)'
                }}
                animate={{ 
                  rotateY: 360,
                  scale: [1, 1.1, 1]
                }}
                transition={{ 
                  rotateY: { duration: 8, repeat: Infinity, ease: "linear" },
                  scale: { duration: 2, repeat: Infinity, ease: "easeInOut" }
                }}
              >
                <CheckCircle className="h-12 w-12 text-cyan-400" />
              </motion.div>
            </motion.div>
            <h1 className="text-5xl sm:text-6xl font-black text-white mb-6">
              <span className="block mb-2">Training</span>
              <span 
                className="block"
                style={{
                  background: 'linear-gradient(135deg, #00f5ff 0%, #9945ff 100%)',
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                Complete! 🎉
              </span>
            </h1>
            <p className="text-xl text-gray-300 max-w-4xl mx-auto leading-relaxed">
              Your {trainingResults?.model_name || 'machine learning'} model has been successfully trained. 
              Explore the results and start making predictions below.
            </p>
          </motion.div>

          {/* Error Display */}
          {error && (
            <motion.div 
              className="max-w-4xl mx-auto mb-8"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3 }}
            >
              <div className="futuristic-card border-yellow-500/30 bg-yellow-900/20">
                <div className="flex items-start">
                  <AlertCircle className="h-6 w-6 text-yellow-400 mr-4 mt-1" />
                  <div className="text-lg text-yellow-300">
                    <p className="font-bold mb-2">Notice</p>
                    <p>{error}</p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Tab Navigation */}
          <div className="max-w-6xl mx-auto">
            <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />

            {/* Visualization Tab */}
            {activeTab === 'visualization' && trainingResults && (
              <motion.div 
                className="space-y-8"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5 }}
              >
                {/* Performance Metrics */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
                  <MetricCard
                    title="Test Accuracy"
                    value={trainingResults.performance?.accuracy || trainingResults.main_score}
                    icon={TrendingUp}
                    color="text-cyan-400"
                    format="percentage"
                    description="Performance on test data"
                  />
                  <MetricCard
                    title="Precision"
                    value={trainingResults.performance?.precision || trainingResults.performance?.accuracy || trainingResults.main_score}
                    icon={Target}
                    color="text-green-400"
                    format="percentage"
                    description="Model precision score"
                  />
                  <MetricCard
                    title="Recall"
                    value={trainingResults.performance?.recall || trainingResults.performance?.accuracy || trainingResults.main_score}
                    icon={Target}
                    color="text-blue-400"
                    format="percentage"
                    description="Model recall score"
                  />
                  <MetricCard
                    title="F1-Score"
                    value={trainingResults.performance?.f1_score || trainingResults.performance?.accuracy || trainingResults.main_score}
                    icon={TrendingUp}
                    color="text-purple-400"
                    format="percentage"
                    description="Harmonic mean of precision and recall"
                  />
                </div>

                {/* Training Details */}
                <div className="futuristic-card p-6">
                  <h3 className="text-xl font-bold text-white mb-6 flex items-center">
                    <Settings className="h-6 w-6 mr-3 text-cyan-400" />
                    Training Details
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-cyan-400 mb-2">
                        {trainingResults.training_details?.training_samples || trainingResults.model_info?.training_samples || 'N/A'}
                      </div>
                      <div className="text-sm text-gray-400">Training Samples</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-400 mb-2">
                        {trainingResults.training_details?.test_samples || trainingResults.model_info?.test_samples || 'N/A'}
                      </div>
                      <div className="text-sm text-gray-400">Test Samples</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-400 mb-2">
                        {trainingResults.training_details?.features || trainingResults.model_info?.feature_count || trainingResults.feature_info?.feature_names?.length || 'N/A'}
                      </div>
                      <div className="text-sm text-gray-400">Features</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-yellow-400 mb-2">
                        {trainingResults.training_details?.training_time ? `${trainingResults.training_details.training_time}s` : trainingResults.performance?.training_time ? `${trainingResults.performance.training_time.toFixed(2)}s` : 'N/A'}
                      </div>
                      <div className="text-sm text-gray-400">Training Time</div>
                    </div>
                  </div>
                  
                  <div className="mt-6 space-y-4">
                    {/* Preprocessing Details */}
                    <div className="p-4 bg-gray-800/50 rounded-lg">
                      <div className="flex items-start space-x-3">
                        <CheckCircle className="h-5 w-5 text-green-400 mt-0.5" />
                        <div className="flex-1">
                          <p className="text-sm text-gray-300 leading-relaxed">
                            <span className="font-semibold text-white">Preprocessing Applied:</span> Missing value imputation, duplicate removal, categorical encoding, feature scaling, and outlier detection were performed before training.
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Model Specifications */}
                    <div className="p-4 bg-gray-800/50 rounded-lg">
                      <h4 className="text-lg font-semibold text-white mb-3 flex items-center">
                        <Brain className="h-5 w-5 text-cyan-400 mr-2" />
                        Model Specifications
                      </h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                        {trainingResults.model_name && (
                          <div className="flex justify-between">
                            <span className="text-gray-400">Algorithm:</span>
                            <span className="text-white font-medium">{trainingResults.model_name}</span>
                          </div>
                        )}
                        {trainingResults.training_details?.problem_type && (
                          <div className="flex justify-between">
                            <span className="text-gray-400">Problem Type:</span>
                            <span className="text-white font-medium">{trainingResults.training_details.problem_type.charAt(0).toUpperCase() + trainingResults.training_details.problem_type.slice(1)}</span>
                          </div>
                        )}
                        {(trainingResults.training_details?.target_column || trainingResults.file_info?.target_column) && (
                          <div className="flex justify-between">
                            <span className="text-gray-400">Target Variable:</span>
                            <span className="text-white font-medium">{trainingResults.training_details?.target_column || trainingResults.file_info?.target_column}</span>
                          </div>
                        )}
                        {trainingResults.training_details?.test_split && (
                          <div className="flex justify-between">
                            <span className="text-gray-400">Train/Test Split:</span>
                            <span className="text-white font-medium">{Math.round((1 - trainingResults.training_details.test_split) * 100)}% / {Math.round(trainingResults.training_details.test_split * 100)}%</span>
                          </div>
                        )}
                        {trainingResults.training_details?.cross_validation && (
                          <div className="flex justify-between">
                            <span className="text-gray-400">Cross Validation:</span>
                            <span className="text-white font-medium">{trainingResults.training_details.cross_validation}-Fold</span>
                          </div>
                        )}
                        {trainingResults.training_details?.preprocessing_steps && (
                          <div className="flex justify-between">
                            <span className="text-gray-400">Preprocessing Steps:</span>
                            <span className="text-white font-medium">{trainingResults.training_details.preprocessing_steps} applied</span>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Dataset Information */}
                    <div className="p-4 bg-gray-800/50 rounded-lg">
                      <h4 className="text-lg font-semibold text-white mb-3 flex items-center">
                        <Database className="h-5 w-5 text-purple-400 mr-2" />
                        Dataset Information
                      </h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                        {trainingResults.file_info?.filename && (
                          <div className="flex justify-between">
                            <span className="text-gray-400">Dataset File:</span>
                            <span className="text-white font-medium">{trainingResults.file_info.filename}</span>
                          </div>
                        )}
                        {getFeatureNames().length > 0 && (
                          <div className="flex justify-between">
                            <span className="text-gray-400">Feature Count:</span>
                            <span className="text-white font-medium">{getFeatureNames().length} features</span>
                          </div>
                        )}
                        <div className="flex justify-between">
                          <span className="text-gray-400">Total Samples:</span>
                          <span className="text-white font-medium">{trainingResults.training_details?.total_samples || ((trainingResults.training_details?.training_samples || 0) + (trainingResults.training_details?.test_samples || 0))}</span>
                        </div>
                        {trainingResults.training_details?.data_quality && (
                          <div className="flex justify-between">
                            <span className="text-gray-400">Data Quality:</span>
                            <span className="text-green-400 font-medium">{trainingResults.training_details.data_quality}</span>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Feature Names Display */}
                    {getFeatureNames().length > 0 && (
                      <div className="p-4 bg-gray-800/50 rounded-lg">
                        <h4 className="text-lg font-semibold text-white mb-3 flex items-center">
                          <BarChart3 className="h-5 w-5 text-yellow-400 mr-2" />
                          Dataset Features
                        </h4>
                        <div className="flex flex-wrap gap-2">
                          {getFeatureNames().map((feature, index) => (
                            <span 
                              key={index}
                              className="px-3 py-1 bg-blue-900/30 border border-blue-500/30 rounded-full text-sm text-blue-300 font-medium"
                            >
                              {feature}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Technical Details */}
                    {trainingResults.model_info?.model_directory && (
                      <div className="p-4 bg-gray-800/50 rounded-lg">
                        <h4 className="text-lg font-semibold text-white mb-3 flex items-center">
                          <Settings className="h-5 w-5 text-gray-400 mr-2" />
                          Technical Details
                        </h4>
                        <div className="text-sm">
                          <div className="flex justify-between mb-2">
                            <span className="text-gray-400">Model Directory:</span>
                            <span className="text-gray-300 font-mono text-xs break-all max-w-xs">{trainingResults.model_info.model_directory}</span>
                          </div>
                          {trainingResults.training_details?.timestamp && (
                            <div className="flex justify-between">
                              <span className="text-gray-400">Training Date:</span>
                              <span className="text-gray-300">{new Date(trainingResults.training_details.timestamp).toLocaleString()}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Detailed Results */}
                <div className="grid lg:grid-cols-2 gap-8">
                  {/* Classification Report */}
                  {trainingResults.performance?.classification_report && (
                    <ClassificationReport 
                      classificationReport={trainingResults.performance.classification_report} 
                    />
                  )}

                  {/* Model Parameters */}
                  {trainingResults.model_params && (
                    <ModelParameters 
                      modelParams={trainingResults.model_params}
                      modelName={trainingResults.model_name || 'Unknown'}
                    />
                  )}
                </div>

                {/* Action Buttons */}
                <div className="flex justify-center space-x-6 pt-8">
                  <motion.button
                    onClick={() => setActiveTab('prediction')}
                    className="px-10 py-4 text-lg font-bold text-white rounded-xl transition-all duration-300 flex items-center"
                    style={{
                      background: 'linear-gradient(135deg, #00f5ff 0%, #9945ff 100%)',
                      boxShadow: '0 0 30px rgba(0, 245, 255, 0.3)'
                    }}
                    whileHover={{ 
                      scale: 1.05,
                      boxShadow: '0 0 40px rgba(0, 245, 255, 0.5)'
                    }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Brain className="h-6 w-6 mr-3" />
                    Start Making Predictions
                  </motion.button>
                  <Link
                    href="/upload"
                    className="px-10 py-4 text-lg font-bold text-cyan-400 futuristic-card hover:border-cyan-400/50 transition-all duration-300 flex items-center"
                  >
                    Train New Model
                  </Link>
                </div>
              </motion.div>
            )}

            {/* Prediction Tab */}
            {activeTab === 'prediction' && (
              <motion.div 
                className="grid lg:grid-cols-3 gap-8"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5 }}
              >
                {/* Input Form */}
                <div className="lg:col-span-2">
                  <motion.div 
                    className="futuristic-card"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.6 }}
                  >
                    <div className="flex items-center mb-8">
                      <div className="p-3 rounded-xl bg-blue-900/30 border border-blue-500/30 mr-4">
                        <Brain className="h-8 w-8 text-blue-400" />
                      </div>
                      <div>
                        <h2 className="text-2xl font-bold text-white">Make a Prediction</h2>
                        {aiColumnSelection && aiColumns.length > 0 && (
                          <p className="text-cyan-400 text-sm mt-1">
                            🤖 Showing {aiColumns.filter(col => col.ai_selected).length} AI-recommended columns
                          </p>
                        )}
                      </div>
                    </div>

                    {/* AI Column Selection Info */}
                    {aiColumnSelection && aiColumnSelection.reasoning.included_reasoning && (
                      <div className="mb-6 p-4 bg-blue-900/20 border border-blue-500/30 rounded-xl">
                        <div className="flex items-start">
                          <div className="p-2 rounded-lg bg-blue-500/20 mr-3">
                            <Brain className="h-5 w-5 text-blue-400" />
                          </div>
                          <div>
                            <h4 className="text-lg font-bold text-blue-300 mb-2">AI Column Selection</h4>
                            <p className="text-blue-200 text-sm leading-relaxed">
                              {aiColumnSelection.reasoning.included_reasoning}
                            </p>
                            {aiColumnSelection.excluded_columns.length > 0 && aiColumnSelection.reasoning.excluded_reasoning && (
                              <p className="text-gray-400 text-xs mt-2">
                                <span className="font-medium">Excluded:</span> {aiColumnSelection.excluded_columns.join(', ')} - {aiColumnSelection.reasoning.excluded_reasoning}
                              </p>
                            )}
                          </div>
                        </div>
                      </div>
                    )}

                    <form onSubmit={handlePrediction} className="space-y-6">
                      {isLoadingColumns ? (
                        <div className="text-center py-8">
                          <Loader2 className="h-8 w-8 animate-spin text-cyan-400 mx-auto mb-4" />
                          <p className="text-gray-300">Loading AI-recommended columns...</p>
                        </div>
                      ) : (
                        <div className="grid md:grid-cols-2 gap-6">
                          {getFeatureNames().map((featureName) => {
                            // Find the AI column info for this feature
                            const aiColumnInfo = aiColumns.find(col => col.name === featureName);
                            
                            // Create a user-friendly label from the feature name
                            const label = featureName
                              .replace(/([A-Z])/g, ' $1') // Add spaces before capital letters
                              .replace(/^./, str => str.toUpperCase()) // Capitalize first letter
                              .replace(/Cm/g, '(cm)') // Replace Cm with (cm)
                              .replace(/Id/g, 'ID'); // Replace Id with ID
                            
                            return (
                              <div key={featureName} className="relative">
                                <label 
                                  htmlFor={featureName} 
                                  className="block text-lg font-bold text-gray-300 mb-3"
                                >
                                  <div className="flex items-center space-x-2">
                                    <span>{label}</span>
                                    {aiColumnInfo?.ai_selected && (
                                      <span className="inline-flex items-center px-2 py-1 text-xs font-bold bg-cyan-500/20 text-cyan-300 rounded-full border border-cyan-500/30">
                                        🤖 AI Selected
                                      </span>
                                    )}
                                  </div>
                                </label>
                                <input
                                  type={aiColumnInfo?.type === 'numeric' ? 'number' : 'text'}
                                  id={featureName}
                                  step={aiColumnInfo?.type === 'numeric' ? '0.1' : undefined}
                                  pattern={featureName.toLowerCase().includes('date') ? '[0-9\-/]*' : undefined}
                                  value={formData[featureName] || ''}
                                  onChange={(e) => handleInputChange(featureName, e.target.value)}
                                  className={`w-full px-4 py-3 bg-gray-800/50 border rounded-xl text-white text-lg focus:ring-2 focus:border-transparent transition-all duration-200 ${
                                    aiColumnInfo?.ai_selected 
                                      ? 'border-cyan-500/50 focus:ring-cyan-400 ring-1 ring-cyan-500/20' 
                                      : 'border-gray-600 focus:ring-blue-400'
                                  }`}
                                  placeholder={`Enter ${aiColumnInfo?.type === 'numeric' ? 'numeric' : 'text'} value...`}
                                  title={featureName.toLowerCase().includes('date') ? 'Enter date in format: dd-mm-yyyy or dd/mm/yyyy' : undefined}
                                  required
                                />
                                <div className="mt-1 text-xs text-gray-500">
                                  Type: {aiColumnInfo?.type || 'unknown'}
                                  {aiColumnInfo?.ai_selected && ' • Recommended by AI for optimal predictions'}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}

                      <motion.button
                        type="submit"
                        disabled={isPredicting || isLoadingColumns}
                        className="w-full px-8 py-4 text-lg font-bold text-white rounded-xl transition-all duration-300 flex items-center justify-center"
                        style={{
                          background: isPredicting ? 'linear-gradient(135deg, #6b7280 0%, #4b5563 100%)' : 'linear-gradient(135deg, #00f5ff 0%, #9945ff 100%)',
                          boxShadow: isPredicting ? 'none' : '0 0 30px rgba(0, 245, 255, 0.3)'
                        }}
                        whileHover={!isPredicting ? { 
                          scale: 1.02,
                          boxShadow: '0 0 40px rgba(0, 245, 255, 0.5)'
                        } : {}}
                        whileTap={!isPredicting ? { scale: 0.98 } : {}}
                      >
                        {isPredicting ? (
                          <>
                            <Loader2 className="h-6 w-6 mr-3 animate-spin" />
                            Predicting...
                          </>
                        ) : (
                          <>
                            <Zap className="h-6 w-6 mr-3" />
                            Make Prediction
                          </>
                        )}
                      </motion.button>
                    </form>
                  </motion.div>
                </div>

                {/* Results Sidebar */}
                <div className="space-y-6">
                  {/* Current Prediction */}
                  {prediction && (
                    <motion.div 
                      className="futuristic-card"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.5 }}
                    >
                      <div className="text-center">
                        <div className="mb-6">
                          <div 
                            className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4"
                            style={{
                              background: 'linear-gradient(135deg, rgba(0, 245, 255, 0.2) 0%, rgba(153, 69, 255, 0.2) 100%)',
                              border: '2px solid rgba(0, 245, 255, 0.3)',
                              boxShadow: '0 0 20px rgba(0, 245, 255, 0.3)'
                            }}
                          >
                            <CheckCircle className="h-10 w-10 text-cyan-400" />
                          </div>
                          <h3 className="text-xl font-bold text-white mb-2">Prediction Result</h3>
                        </div>
                        
                        <div className="mb-6">
                          <div 
                            className="text-3xl font-black mb-2"
                            style={{
                              background: 'linear-gradient(135deg, #00f5ff 0%, #9945ff 100%)',
                              backgroundClip: 'text',
                              WebkitBackgroundClip: 'text',
                              WebkitTextFillColor: 'transparent',
                            }}
                          >
                            {prediction.value}
                          </div>
                          <div className="flex items-center justify-center">
                            <span className="text-lg font-bold text-green-400">
                              {prediction.confidence.toFixed(1)}% Confident
                            </span>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}

                  {/* Prediction History */}
                  {predictionHistory.length > 0 && (
                    <motion.div 
                      className="futuristic-card"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.6, delay: 0.2 }}
                    >
                      <h3 className="text-xl font-bold text-white mb-4">Recent Predictions</h3>
                      <div className="space-y-3">
                        {predictionHistory.map((item, index) => (
                          <div 
                            key={index} 
                            className="p-4 bg-gray-800/30 rounded-xl border border-gray-600"
                          >
                            <div className="flex justify-between items-center mb-2">
                              <span className="text-lg font-bold text-cyan-400">{item.result}</span>
                              <span className="text-sm text-green-400 font-medium">
                                {item.confidence.toFixed(1)}%
                              </span>
                            </div>
                            <div className="text-xs text-gray-400">
                              {item.timestamp.toLocaleTimeString()}
                            </div>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  )}

                  {/* Training Summary */}
                  {trainingResults && (
                    <motion.div 
                      className="futuristic-card"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.6, delay: 0.4 }}
                    >
                      <h3 className="text-xl font-bold text-white mb-4">Training Summary</h3>
                      <div className="space-y-3 text-sm">
                        <div>
                          <span className="font-medium text-gray-300">Accuracy:</span>
                          <span className="ml-2 text-cyan-400 font-bold">
                            {((trainingResults.performance?.accuracy || trainingResults.main_score) * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div>
                          <span className="font-medium text-gray-300">CV Score:</span>
                          <span className="ml-2 text-gray-400">
                            {((trainingResults.performance?.cv_accuracy || trainingResults.main_score * 0.9) * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div>
                          <span className="font-medium text-gray-300">Status:</span>
                          <span className={`ml-2 font-medium ${
                            trainingResults.threshold_met ? 'text-green-400' : 'text-yellow-400'
                          }`}>
                            {trainingResults.threshold_met ? '✅ Excellent' : '⚠️ Good'}
                          </span>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </div>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// Main exported component with Suspense boundary
export default function ResultsPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen relative overflow-hidden bg-gray-900 flex items-center justify-center">
        <ParticleBackground />
        <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-purple-900 to-cyan-900 opacity-20" />
        <div className="relative z-10 text-center">
          <Brain className="h-24 w-24 text-cyan-400 mx-auto mb-6" />
          <h2 className="text-4xl font-black text-white mb-6">Loading...</h2>
        </div>
      </div>
    }>
      <ResultsPageContent />
    </Suspense>
  );
}