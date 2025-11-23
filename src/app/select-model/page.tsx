'use client';

import { useState, useEffect, useRef } from 'react';
import { CheckCircle, Brain, Target, Loader2, AlertCircle, Zap, Activity, TrendingUp, Star, Database } from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { apiClient, ModelRecommendation } from '@/lib/api';

/**
 * Futuristic Model Card Component with holographic effects
 */
interface ModelCardProps {
  name: string;
  description: string;
  accuracy: number;
  isRecommended?: boolean;
  onSelect: () => void;
  isSelected: boolean;
  index: number;
}

function ModelCard({ name, description, accuracy, isRecommended, onSelect, isSelected, index }: ModelCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ 
        duration: 0.6, 
        delay: index * 0.1,
        ease: "easeOut"
      }}
      whileHover={{ 
        y: -10,
        scale: 1.02,
        transition: { duration: 0.3 }
      }}
      whileTap={{ scale: 0.98 }}
      className={`
        relative futuristic-card cursor-pointer group transition-all duration-500 
        ${isSelected ? 'ring-2 ring-cyan-400' : ''} 
        ${isRecommended ? 'ring-2 ring-purple-400' : ''}
      `}
      style={{
        background: isSelected 
          ? 'linear-gradient(135deg, rgba(0, 245, 255, 0.15) 0%, rgba(153, 69, 255, 0.15) 100%)'
          : isRecommended
          ? 'linear-gradient(135deg, rgba(153, 69, 255, 0.1) 0%, rgba(255, 0, 128, 0.1) 100%)'
          : 'rgba(255, 255, 255, 0.05)',
        backdropFilter: 'blur(20px)',
        border: `1px solid ${
          isSelected ? 'rgba(0, 245, 255, 0.3)' : 
          isRecommended ? 'rgba(153, 69, 255, 0.3)' : 
          'rgba(255, 255, 255, 0.1)'
        }`
      }}
      onClick={onSelect}
    >
      {/* Recommended Badge */}
      <AnimatePresence>
        {isRecommended && (
          <motion.div
            initial={{ opacity: 0, scale: 0, rotate: -180 }}
            animate={{ opacity: 1, scale: 1, rotate: 0 }}
            transition={{ type: "spring", stiffness: 300, delay: 0.3 }}
            className="absolute -top-0 -right-1 z-10"
          >
            <div className="flex items-center bg-gradient-to-r from-purple-600 to-pink-600 text-white px-3 py-1 rounded-full text-xs font-bold shadow-lg">
              <Star className="h-3 w-8 mr-1" />
              AI CHOICE
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Holographic Corner Effect */}
      <div className="absolute top-0 right-0 w-20 h-20 opacity-20 pointer-events-none">
        <div 
          className="w-full h-full"
          style={{
            background: 'linear-gradient(45deg, transparent 30%, rgba(0, 245, 255, 0.5) 50%, transparent 70%)',
            clipPath: 'polygon(0 0, 100% 0, 100% 100%)'
          }}
        />
      </div>

      {/* Header Section */}
      <div className="flex items-start justify-between mb-4">
        <motion.div
          className="flex items-center group-hover:scale-105 transition-transform duration-300"
        >
          <motion.div
            className="relative mr-3"
            animate={isSelected ? {
              boxShadow: [
                '0 0 10px rgba(0, 245, 255, 0.5)',
                '0 0 20px rgba(0, 245, 255, 0.8)',
                '0 0 10px rgba(0, 245, 255, 0.5)'
              ]
            } : {}}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            <div 
              className="w-12 h-12 rounded-lg flex items-center justify-center"
              style={{
                background: 'linear-gradient(135deg, rgba(0, 245, 255, 0.2) 0%, rgba(153, 69, 255, 0.2) 100%)',
                border: '1px solid rgba(0, 245, 255, 0.3)'
              }}
            >
              <Brain className="h-6 w-6 text-cyan-400" />
            </div>
          </motion.div>
          <div>
            <h3 className="text-xl font-bold text-white mb-1">{name}</h3>
            <div className="flex items-center space-x-2 text-xs">
              <Activity className="h-3 w-3 text-green-400" />
              <span className="text-green-400 font-medium">Active</span>
            </div>
          </div>
        </motion.div>

        {/* Accuracy Badge */}
        <motion.div
          className="flex flex-col items-center"
          whileHover={{ scale: 1.1 }}
        >
          <div 
            className="w-16 h-16 rounded-full flex items-center justify-center text-sm font-bold text-white relative"
            style={{
              background: `conic-gradient(from 0deg, #00f5ff 0deg, #9945ff ${accuracy * 3.6}deg, rgba(255,255,255,0.1) ${accuracy * 3.6}deg)`
            }}
          >
            <div 
              className="w-24 h-16 rounded-full flex items-center justify-center"
              style={{ background: 'rgba(10, 10, 15, 0.9)' }}
            >
              {accuracy}
            </div>
          </div>
          <span className="text-xs text-gray-400 mt-1">Accuracy</span>
        </motion.div>
      </div>

      {/* Description */}
      <p className="text-gray-300 text-sm leading-relaxed mb-6">
        {description}
      </p>

      {/* Footer */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-2 text-xs">
          <TrendingUp className="h-3 w-3 text-blue-400" />
          <span className="text-blue-400">High Performance</span>
        </div>
        
        <motion.button
          className={`
            px-4 py-2 text-sm font-bold rounded-lg transition-all duration-300 
            ${isSelected
              ? 'bg-gradient-to-r from-cyan-500 to-purple-500 text-white shadow-lg'
              : 'bg-gray-700/50 text-gray-300 hover:bg-gray-600/50'
            }
          `}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {isSelected ? (
            <span className="flex items-center">
              <CheckCircle className="h-4 w-4 mr-1" />
              Selected
            </span>
          ) : (
            'Select Model'
          )}
        </motion.button>
      </div>
    </motion.div>
  );
}

/**
 * Futuristic Model Selection Page
 */
export default function SelectModelPage() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);
  const [isTraining, setIsTraining] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [recommendations, setRecommendations] = useState<ModelRecommendation | null>(null);
  const [error, setError] = useState<string>('');
  const [fileId, setFileId] = useState<string>('');
  
  // Prevent duplicate requests in React development mode
  const hasRequestedRef = useRef(false);

  // Load file ID from localStorage and fetch recommendations
  useEffect(() => {
    const loadRecommendations = async () => {
      // Prevent duplicate requests
      if (hasRequestedRef.current) {
        console.log('⚠️  Request already made, skipping duplicate');
        return;
      }
      hasRequestedRef.current = true;

      const storedFileId = localStorage.getItem('currentFileId');
      
      if (!storedFileId) {
        setError('No file uploaded. Please upload a dataset first.');
        setIsLoading(false);
        return;
      }

      setFileId(storedFileId);

      try {
        console.log('🚀 Making single request to get model recommendations');
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

  const handleStartTraining = async () => {
    if (!selectedModel || !fileId) {
      setError('Please select a model first');
      return;
    }

    // Find the selected model details
    const allModels = [
      ...(recommendations?.recommendations?.recommended_models || []),
      ...(recommendations?.recommendations?.alternative_models || [])
    ];
    const selectedModelData = allModels.find(model => 
      model.name === selectedModel || 
      model.name.toLowerCase().replace(/\s+/g, '-') === selectedModel
    );
    const modelName = selectedModelData?.name || selectedModel;

    setIsTraining(true);
    setError('');

    try {
      console.log('Starting training for:', modelName);
      
      const response = await apiClient.startTraining(fileId, modelName);
      
      if (response.success && response.result) {
        const completeResults = {
          ...response.result,
          feature_info: response.feature_info
        };
        localStorage.setItem('trainingResults', JSON.stringify(completeResults));
        
        router.push('/results');
      } else {
        throw new Error(response.error || 'Training failed');
      }
    } catch (err) {
      console.error('Training error:', err);
      setError('Training failed. Please try again.');
      setIsTraining(false);
    }
  };

  // Fallback models for when LLM is not available
  const fallbackModels = [
    {
      id: 'random-forest',
      name: 'Random Forest',
      description: 'Ensemble learning method that constructs multiple decision trees. Perfect for complex patterns and high accuracy.',
      accuracy: 87,
      isRecommended: true,
    },
    {
      id: 'svm',
      name: 'Support Vector Machine',
      description: 'Advanced algorithm for classification and regression. Excels with high-dimensional data and complex boundaries.',
      accuracy: 84,
      isRecommended: false,
    },
    {
      id: 'gradient-boosting',
      name: 'Gradient Boosting',
      description: 'Sequential ensemble method building models iteratively. State-of-the-art performance for structured data.',
      accuracy: 89,
      isRecommended: false,
    },
    {
      id: 'neural-network',
      name: 'Neural Network',
      description: 'Deep learning approach mimicking brain neurons. Excellent for pattern recognition and complex relationships.',
      accuracy: 91,
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
        isRecommended: index === 0,
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
      <div className="min-h-screen flex items-center justify-center relative overflow-hidden pt-16">
        {/* Animated Background */}
        <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-purple-900 to-cyan-900 opacity-20" />
        <div className="absolute inset-0 geometric-pattern opacity-20" />
        
        <motion.div 
          className="text-center relative z-10"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex justify-center mb-8">
            <div className="w-20 h-20 border-4 border-cyan-400/30 border-t-cyan-400 rounded-full animate-spin" />
          </div>
          
          <motion.h2 
            className="text-3xl font-bold text-white mb-4"
            animate={{ opacity: [0.7, 1, 0.7] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            Analyzing Your Data
          </motion.h2>
          
          <p className="text-xl text-gray-300 max-w-md mx-auto mb-8">
            Our AI is examining your dataset to recommend the perfect machine learning models
          </p>
          
          <motion.div 
            className="glass-effect rounded-lg p-6 max-w-md mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <div className="space-y-3 text-sm text-gray-400">
              <motion.div 
                className="flex items-center"
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 1.5, repeat: Infinity, delay: 0 }}
              >
                <CheckCircle className="h-4 w-4 text-green-400 mr-2" />
                Data structure analysis complete
              </motion.div>
              <motion.div 
                className="flex items-center"
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 1.5, repeat: Infinity, delay: 0.5 }}
              >
                <CheckCircle className="h-4 w-4 text-green-400 mr-2" />
                Feature correlation calculated
              </motion.div>
              <motion.div 
                className="flex items-center text-cyan-400"
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 1.5, repeat: Infinity, delay: 1 }}
              >
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
                Evaluating model compatibility...
              </motion.div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center relative overflow-hidden" style={{paddingTop: '116px'}}>
        <div className="absolute inset-0 bg-gradient-to-br from-red-900/20 via-gray-900 to-gray-900" />
        
        <motion.div 
          className="max-w-md w-full glass-effect rounded-lg p-8 text-center relative z-10"
          initial={{ opacity: 0, scale: 0.9, y: 50 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <motion.div
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <AlertCircle className="h-16 w-16 text-red-400 mx-auto mb-6" />
          </motion.div>
          
          <h2 className="text-2xl font-bold text-white mb-4">
            Error Loading Recommendations
          </h2>
          <p className="text-gray-300 mb-8">{error}</p>
          
          <div className="space-y-3">
            <Link href="/upload">
              <motion.button
                className="btn-primary w-full"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Upload New Dataset
              </motion.button>
            </Link>
            <motion.button
              onClick={() => window.location.reload()}
              className="w-full px-4 py-2 text-sm font-medium text-cyan-400 glass-effect border border-cyan-500/30 rounded-lg hover:bg-cyan-500/10 transition-all duration-300"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              Retry Analysis
            </motion.button>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen relative overflow-hidden" style={{paddingTop: '60px'}}>
      {/* Animated Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-purple-900 to-cyan-900 opacity-20" />
      <div className="absolute inset-0 geometric-pattern opacity-20" />
      
      {/* Main Content with proper spacing */}
      <div className="relative z-10  pb-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Header */}
          <motion.div 
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <motion.div 
              className="flex justify-center mb-8"
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
            >
              <Target className="h-16 w-16 text-cyan-400" style={{ filter: 'drop-shadow(0 0 20px rgba(0, 245, 255, 0.5))' }} />
            </motion.div>
            
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black text-white mb-6">
              <span className="gradient-text">AI Model</span> Selection
            </h1>
            <p className="text-lg sm:text-xl text-gray-300 max-w-4xl mx-auto px-4">
              Our advanced AI has analyzed your dataset and selected the most optimal models for your specific use case
            </p>
            
            {/* Dataset Info */}
            {recommendations?.dataset_info && (
              <motion.div 
                className="mt-8 glass-effect rounded-xl p-6 max-w-5xl mx-auto"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                <h3 className="font-bold text-white mb-6 flex items-center justify-center text-lg">
                  <Database className="h-5 w-5 mr-2 text-cyan-400" />
                  Dataset Analysis
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-sm">
                  {[
                    { label: "Total Rows", value: recommendations.dataset_info.total_rows, icon: "📊" },
                    { label: "Features", value: recommendations.dataset_info.total_columns, icon: "🔢" },
                    { label: "Numeric", value: recommendations.dataset_info.numeric_columns, icon: "📈" },
                    { label: "Categorical", value: recommendations.dataset_info.categorical_columns, icon: "🏷️" }
                  ].map((stat, index) => (
                    <motion.div
                      key={stat.label}
                      className="text-center"
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 0.5 + index * 0.1 }}
                    >
                      <div className="text-2xl mb-2">{stat.icon}</div>
                      <div className="text-2xl font-bold text-cyan-400 mb-1">{stat.value}</div>
                      <div className="text-gray-400">{stat.label}</div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}
          </motion.div>

          {/* Recommended Model Section */}
          {recommendedModel && (
            <motion.div 
              className="mb-16"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <div className="text-center mb-8">
                <h2 className="text-2xl sm:text-3xl font-bold text-white mb-4 flex items-center justify-center">
                  <Star className="h-7 w-7 sm:h-8 sm:w-8 text-purple-400 mr-3" />
                  AI Recommended Model
                </h2>
                <p className="text-gray-300">
                  This model has the highest compatibility score with your data
                </p>
              </div>
              
              <div className="max-w-3xl mx-auto">
                <ModelCard
                  name={recommendedModel.name}
                  description={recommendedModel.description}
                  accuracy={recommendedModel.accuracy}
                  isRecommended={true}
                  onSelect={() => handleModelSelect(recommendedModel.id)}
                  isSelected={selectedModel === recommendedModel.id}
                  index={0}
                />
              </div>
            </motion.div>
          )}

          {/* Alternative Models Section */}
          {otherModels.length > 0 && (
            <motion.div 
              className="mb-16"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 }}
            >
              <div className="text-center mb-8">
                <h2 className="text-2xl sm:text-3xl font-bold text-white mb-4">
                  Alternative Models
                </h2>
                <p className="text-gray-300">
                  Other powerful models that could work well with your dataset
                </p>
              </div>
              
              <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-6">
                {otherModels.map((model, index) => (
                  <ModelCard
                    key={model.id}
                    name={model.name}
                    description={model.description}
                    accuracy={model.accuracy}
                    isRecommended={false}
                    onSelect={() => handleModelSelect(model.id)}
                    isSelected={selectedModel === model.id}
                    index={index + 1}
                  />
                ))}
              </div>
            </motion.div>
          )}

          {/* Action Buttons */}
          <motion.div 
            className="flex flex-col sm:flex-row justify-center gap-4 px-4"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1 }}
          >
            <Link href="/upload">
              <motion.button
                className="glass-effect px-6 sm:px-8 py-4 text-base sm:text-lg font-semibold text-gray-300 border border-gray-500/30 rounded-lg hover:bg-gray-700/30 transition-all duration-300 w-full sm:w-auto min-w-[220px]"
                whileHover={{ scale: 1.02, borderColor: 'rgba(156, 163, 175, 0.5)' }}
                whileTap={{ scale: 0.98 }}
              >
                Upload Different Dataset
              </motion.button>
            </Link>
            
            <motion.button
              onClick={handleStartTraining}
              disabled={!selectedModel || isTraining}
              className={`
                px-6 sm:px-8 py-4 text-base sm:text-lg font-bold rounded-lg transition-all duration-300 w-full sm:w-auto min-w-[220px]
                ${selectedModel && !isTraining
                  ? 'btn-primary'
                  : 'bg-gray-700/50 text-gray-500 cursor-not-allowed'
                }
              `}
              whileHover={selectedModel && !isTraining ? { 
                scale: 1.05,
                boxShadow: '0 0 30px rgba(0, 245, 255, 0.4)'
              } : {}}
              whileTap={selectedModel && !isTraining ? { scale: 0.98 } : {}}
            >
              {isTraining ? (
                <span className="flex items-center justify-center">
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                  Training Model...
                </span>
              ) : (
                <span className="flex items-center justify-center">
                  <Zap className="mr-2 h-5 w-5" />
                  Start Training
                </span>
              )}
            </motion.button>
          </motion.div>
        </div>
      </div>
    </div>
  );
}