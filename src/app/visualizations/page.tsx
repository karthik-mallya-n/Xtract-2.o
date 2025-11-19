'use client';

import { useState, useEffect, useRef } from 'react';
import { BarChart3, TrendingUp, Activity, Eye, Loader2, AlertCircle, Download, Settings } from 'lucide-react';
import Link from 'next/link';
import { motion } from 'framer-motion';

/**
 * Comprehensive Data Visualization Page
 * Supports all major visualization types for uploaded datasets
 */

interface ColumnInfo {
  type: string;
  category: 'categorical' | 'continuous' | 'datetime';
  non_null_count: number;
  null_count: number;
  unique_count: number;
  sample_values: (string | number | Date)[];
}

interface VisualizationAnalysis {
  available_visualizations: {
    univariate: {
      categorical: string[];
      continuous: string[];
    };
    bivariate: {
      continuous_vs_continuous: string[];
      categorical_vs_continuous: string[];
      categorical_vs_categorical: string[];
    };
    multivariate: string[];
    time_series: string[];
    distribution: string[];
  };
  column_analysis: {
    total_rows: number;
    total_columns: number;
    columns: Record<string, ColumnInfo>;
  };
  categorical_columns: string[];
  continuous_columns: string[];
  datetime_columns: string[];
}

interface VisualizationParams {
  column?: string;
  x_column?: string;
  y_column?: string;
  z_column?: string;
  color_column?: string;
  bins?: number;
}

interface VisualizationResult {
  success: boolean;
  html?: string;
  type: string;
  error?: string;
  [key: string]: string | number | boolean | undefined;
}

/*
 * Visualization Control Panel Component
 */
interface ControlPanelProps {
  analysis: VisualizationAnalysis;
  onCreateVisualization: (type: string, params: VisualizationParams) => void;
  isLoading: boolean;
}

function VisualizationControlPanel({ analysis, onCreateVisualization, isLoading }: ControlPanelProps) {
  const [selectedType, setSelectedType] = useState<string>('');
  const [params, setParams] = useState<VisualizationParams>({});
  const [activeCategory, setActiveCategory] = useState<string>('univariate');

  const handleCreateVisualization = () => {
    if (selectedType) {
      onCreateVisualization(selectedType, params);
    }
  };

  const renderParameterInputs = () => {
    if (!selectedType) return null;

    const inputs = [];

    // Common parameters based on visualization type
    if (['histogram', 'bar_chart', 'pie_chart', 'density_plot'].includes(selectedType)) {
      inputs.push(
        <div key="column" className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">Column</label>
          <select
            value={params.column || ''}
            onChange={(e) => setParams({...params, column: e.target.value})}
            className="w-full bg-gray-800 text-white border border-gray-600 rounded-lg px-3 py-2"
          >
            <option value="">Select Column</option>
            {selectedType === 'bar_chart' || selectedType === 'pie_chart' 
              ? analysis.categorical_columns.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))
              : analysis.continuous_columns.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))
            }
          </select>
        </div>
      );

      if (selectedType === 'histogram') {
        inputs.push(
          <div key="bins" className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">Number of Bins</label>
            <input
              type="number"
              value={params.bins || 30}
              onChange={(e) => setParams({...params, bins: parseInt(e.target.value)})}
              className="w-full bg-gray-800 text-white border border-gray-600 rounded-lg px-3 py-2"
              min="10"
              max="100"
            />
          </div>
        );
      }
    }

    if (['scatter_plot', 'line_chart'].includes(selectedType)) {
      inputs.push(
        <div key="x_column" className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">X Column</label>
          <select
            value={params.x_column || ''}
            onChange={(e) => setParams({...params, x_column: e.target.value})}
            className="w-full bg-gray-800 text-white border border-gray-600 rounded-lg px-3 py-2"
          >
            <option value="">Select X Column</option>
            {analysis.continuous_columns.map(col => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
      );

      inputs.push(
        <div key="y_column" className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">Y Column</label>
          <select
            value={params.y_column || ''}
            onChange={(e) => setParams({...params, y_column: e.target.value})}
            className="w-full bg-gray-800 text-white border border-gray-600 rounded-lg px-3 py-2"
          >
            <option value="">Select Y Column</option>
            {analysis.continuous_columns.map(col => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
      );

      inputs.push(
        <div key="color_column" className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">Color By (Optional)</label>
          <select
            value={params.color_column || ''}
            onChange={(e) => setParams({...params, color_column: e.target.value})}
            className="w-full bg-gray-800 text-white border border-gray-600 rounded-lg px-3 py-2"
          >
            <option value="">No Color Grouping</option>
            {analysis.categorical_columns.map(col => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
      );
    }

    if (['box_plot', 'violin_plot'].includes(selectedType)) {
      inputs.push(
        <div key="y_column" className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">Value Column</label>
          <select
            value={params.y_column || ''}
            onChange={(e) => setParams({...params, y_column: e.target.value})}
            className="w-full bg-gray-800 text-white border border-gray-600 rounded-lg px-3 py-2"
          >
            <option value="">Select Column</option>
            {analysis.continuous_columns.map(col => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
      );

      inputs.push(
        <div key="x_column" className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">Group By (Optional)</label>
          <select
            value={params.x_column || ''}
            onChange={(e) => setParams({...params, x_column: e.target.value})}
            className="w-full bg-gray-800 text-white border border-gray-600 rounded-lg px-3 py-2"
          >
            <option value="">No Grouping</option>
            {analysis.categorical_columns.map(col => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
      );
    }

    if (['heatmap_crosstab'].includes(selectedType)) {
      inputs.push(
        <div key="x_column" className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">X Column</label>
          <select
            value={params.x_column || ''}
            onChange={(e) => setParams({...params, x_column: e.target.value})}
            className="w-full bg-gray-800 text-white border border-gray-600 rounded-lg px-3 py-2"
          >
            <option value="">Select X Column</option>
            {analysis.categorical_columns.map(col => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
      );

      inputs.push(
        <div key="y_column" className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">Y Column</label>
          <select
            value={params.y_column || ''}
            onChange={(e) => setParams({...params, y_column: e.target.value})}
            className="w-full bg-gray-800 text-white border border-gray-600 rounded-lg px-3 py-2"
          >
            <option value="">Select Y Column</option>
            {analysis.categorical_columns.map(col => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
      );
    }

    if (['scatter_3d'].includes(selectedType)) {
      inputs.push(
        <div key="x_column" className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">X Column</label>
          <select
            value={params.x_column || ''}
            onChange={(e) => setParams({...params, x_column: e.target.value})}
            className="w-full bg-gray-800 text-white border border-gray-600 rounded-lg px-3 py-2"
          >
            <option value="">Select X Column</option>
            {analysis.continuous_columns.map(col => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
      );

      inputs.push(
        <div key="y_column" className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">Y Column</label>
          <select
            value={params.y_column || ''}
            onChange={(e) => setParams({...params, y_column: e.target.value})}
            className="w-full bg-gray-800 text-white border border-gray-600 rounded-lg px-3 py-2"
          >
            <option value="">Select Y Column</option>
            {analysis.continuous_columns.map(col => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
      );

      inputs.push(
        <div key="z_column" className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">Z Column</label>
          <select
            value={params.z_column || ''}
            onChange={(e) => setParams({...params, z_column: e.target.value})}
            className="w-full bg-gray-800 text-white border border-gray-600 rounded-lg px-3 py-2"
          >
            <option value="">Select Z Column</option>
            {analysis.continuous_columns.map(col => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
      );
    }

    return inputs;
  };

  const getVisualizationsForCategory = (category: string) => {
    switch (category) {
      case 'univariate':
        return [
          ...analysis.available_visualizations.univariate.categorical.map(viz => ({
            id: viz,
            name: viz.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
            category: 'Categorical'
          })),
          ...analysis.available_visualizations.univariate.continuous.map(viz => ({
            id: viz,
            name: viz.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
            category: 'Continuous'
          }))
        ];
      case 'bivariate':
        return [
          ...analysis.available_visualizations.bivariate.continuous_vs_continuous.map(viz => ({
            id: viz,
            name: viz.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
            category: 'Continuous vs Continuous'
          })),
          ...analysis.available_visualizations.bivariate.categorical_vs_continuous.map(viz => ({
            id: viz,
            name: viz.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
            category: 'Categorical vs Continuous'
          })),
          ...analysis.available_visualizations.bivariate.categorical_vs_categorical.map(viz => ({
            id: viz,
            name: viz.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
            category: 'Categorical vs Categorical'
          }))
        ];
      case 'multivariate':
        return analysis.available_visualizations.multivariate.map(viz => ({
          id: viz,
          name: viz.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
          category: 'Multivariate'
        }));
      case 'distribution':
        return analysis.available_visualizations.distribution.map(viz => ({
          id: viz,
          name: viz.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
          category: 'Distribution'
        }));
      default:
        return [];
    }
  };

  return (
    <div className="glass-effect rounded-xl p-6">
      <div className="flex items-center mb-6">
        <Settings className="h-6 w-6 text-cyan-400 mr-3" />
        <h3 className="text-xl font-bold text-white">Visualization Controls</h3>
      </div>

      {/* Category Selection */}
      <div className="mb-6">
        <div className="flex flex-wrap gap-2">
          {['univariate', 'bivariate', 'multivariate', 'distribution'].map(category => (
            <motion.button
              key={category}
              onClick={() => {
                setActiveCategory(category);
                setSelectedType('');
                setParams({});
              }}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
                activeCategory === category
                  ? 'bg-cyan-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {category.charAt(0).toUpperCase() + category.slice(1)}
            </motion.button>
          ))}
        </div>
      </div>

      {/* Visualization Type Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-300 mb-3">Visualization Type</label>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-48 overflow-y-auto">
          {getVisualizationsForCategory(activeCategory).map(viz => (
            <motion.button
              key={viz.id}
              onClick={() => {
                setSelectedType(viz.id);
                setParams({});
              }}
              className={`text-left p-3 rounded-lg transition-all duration-300 ${
                selectedType === viz.id
                  ? 'bg-purple-500/20 border border-purple-400 text-purple-300'
                  : 'bg-gray-700/50 text-gray-300 hover:bg-gray-600/50'
              }`}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="font-medium">{viz.name}</div>
              <div className="text-xs text-gray-400">{viz.category}</div>
            </motion.button>
          ))}
        </div>
      </div>

      {/* Parameters */}
      {selectedType && (
        <motion.div 
          className="mb-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <label className="block text-sm font-medium text-gray-300 mb-3">Parameters</label>
          {renderParameterInputs()}
        </motion.div>
      )}

      {/* Create Button */}
      <motion.button
        onClick={handleCreateVisualization}
        disabled={!selectedType || isLoading}
        className={`w-full py-3 rounded-lg font-medium transition-all duration-300 ${
          selectedType && !isLoading
            ? 'bg-gradient-to-r from-cyan-500 to-purple-500 text-white hover:shadow-lg'
            : 'bg-gray-700 text-gray-500 cursor-not-allowed'
        }`}
        whileHover={selectedType && !isLoading ? { scale: 1.02 } : {}}
        whileTap={selectedType && !isLoading ? { scale: 0.98 } : {}}
      >
        {isLoading ? (
          <span className="flex items-center justify-center">
            <Loader2 className="h-4 w-4 animate-spin mr-2" />
            Creating Visualization...
          </span>
        ) : (
          <span className="flex items-center justify-center">
            <Eye className="h-4 w-4 mr-2" />
            Create Visualization
          </span>
        )}
      </motion.button>
    </div>
  );
}

export default function VisualizationPage() {
  const [analysis, setAnalysis] = useState<VisualizationAnalysis | null>(null);
  const [currentVisualization, setCurrentVisualization] = useState<VisualizationResult | null>(null);
  const [isLoadingAnalysis, setIsLoadingAnalysis] = useState(true);
  const [isCreatingVisualization, setIsCreatingVisualization] = useState(false);
  const [error, setError] = useState<string>('');
  const [fileId, setFileId] = useState<string>('');

  // Prevent duplicate requests
  const hasRequestedRef = useRef(false);

  useEffect(() => {
    const loadAnalysis = async () => {
      if (hasRequestedRef.current) {
        console.log('⚠️  Analysis request already made, skipping duplicate');
        return;
      }
      hasRequestedRef.current = true;

      const storedFileId = localStorage.getItem('currentFileId');
      
      if (!storedFileId) {
        console.log('⚠️  No file uploaded, using sample data for demo');
        // Use sample data for demo purposes
        setFileId('sample');
        setError('');
      } else {
        setFileId(storedFileId);
      }

      try {
        console.log('🚀 Analyzing dataset for visualizations');
        const currentFileId = storedFileId || 'sample';
        console.log(`📊 Using file_id: ${currentFileId}`);
        const response = await fetch(`http://localhost:5000/api/visualizations/analyze?file_id=${currentFileId}`);
        const result = await response.json();
        
        console.log('📊 Analysis result:', result);
        
        if (result.success) {
          setAnalysis(result.analysis);
          console.log('✅ Analysis set successfully');
        } else {
          setError(result.error || 'Failed to analyze dataset');
        }
      } catch (err) {
        console.error('Error analyzing dataset:', err);
        setError('Failed to connect to the backend. Please ensure the Flask server is running.');
      } finally {
        setIsLoadingAnalysis(false);
      }
    };

    loadAnalysis();
  }, []);

  const handleCreateVisualization = async (type: string, params: VisualizationParams) => {
    setIsCreatingVisualization(true);
    setError('');
    setCurrentVisualization(null); // Clear previous visualization

    try {
      console.log('🎨 Creating visualization:', { type, params, fileId });
      
      const response = await fetch('http://localhost:5000/api/visualizations/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_id: fileId,
          visualization_type: type,
          parameters: params
        })
      });
      const result = await response.json();
      console.log('📊 Visualization response:', result);

      if (result.success && result.visualization) {
        
        setCurrentVisualization(result.visualization);
        console.log('✅ Visualization set successfully');
      } else {
        setError(result.error || 'Failed to create visualization');
        console.error('❌ Visualization creation failed:', result.error);
      }
    } catch (err) {
      console.error('Error creating visualization:', err);
      setError('Failed to create visualization. Please check if the backend server is running.');
    } finally {
      setIsCreatingVisualization(false);
    }
  };

  const handleExportVisualization = () => {
    if (currentVisualization?.html) {
      // Create a downloadable HTML file
      const blob = new Blob([currentVisualization.html], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `visualization_${Date.now()}.html`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  if (isLoadingAnalysis) {
    return (
      <div className="min-h-screen flex items-center justify-center relative overflow-hidden pt-16">
        <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-purple-900 to-cyan-900 opacity-20" />
        <div className="absolute inset-0 geometric-pattern opacity-20" />
        
        <motion.div 
          className="text-center relative z-10"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6 }}
        >
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
            className="mx-auto mb-8"
          >
            <div className="w-20 h-20 border-4 border-cyan-400/30 border-t-cyan-400 rounded-full" />
          </motion.div>
          
          <motion.h2 
            className="text-3xl font-bold text-white mb-4"
            animate={{ opacity: [0.7, 1, 0.7] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            Analyzing Dataset Structure
          </motion.h2>
          
          <p className="text-xl text-gray-300 max-w-md mx-auto mb-8">
            Examining your data to determine the best visualization options
          </p>
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
            Error Loading Dataset
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
    <div className="min-h-screen relative overflow-hidden" style={{paddingTop: '50px', paddingBottom: '48px'}}>
      <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-purple-900 to-cyan-900 opacity-20" />
      <div className="absolute inset-0 geometric-pattern opacity-20" />
      
      <div className="relative z-10">
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
              animate={{ rotateY: 360 }}
              transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
            >
              <BarChart3 className="h-16 w-16 text-cyan-400" style={{ filter: 'drop-shadow(0 0 20px rgba(0, 245, 255, 0.5))' }} />
            </motion.div>
            
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black text-white mb-6">
              <span className="gradient-text">Data</span> Visualization
            </h1>
            <p className="text-lg sm:text-xl text-gray-300 max-w-4xl mx-auto px-4">
              Explore your dataset with comprehensive visualizations. Create charts, plots, and graphs to understand your data better.
            </p>

            {/* Dataset Info */}
            {analysis && (
              <motion.div 
                className="mt-8 glass-effect rounded-xl p-6 max-w-5xl mx-auto"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                <h3 className="font-bold text-white mb-6 flex items-center justify-center text-lg">
                  <Activity className="h-5 w-5 mr-2 text-cyan-400" />
                  Dataset Overview
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-sm">
                  {[
                    { label: "Total Rows", value: analysis.column_analysis.total_rows, icon: "📊" },
                    { label: "Columns", value: analysis.column_analysis.total_columns, icon: "🔢" },
                    { label: "Categorical", value: analysis.categorical_columns.length, icon: "🏷️" },
                    { label: "Continuous", value: analysis.continuous_columns.length, icon: "📈" }
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

          <div className="grid lg:grid-cols-3 gap-8">
            {/* Control Panel */}
            <div className="lg:col-span-1">
              {analysis && (
                <VisualizationControlPanel
                  analysis={analysis}
                  onCreateVisualization={handleCreateVisualization}
                  isLoading={isCreatingVisualization}
                />
              )}
            </div>

            {/* Visualization Display */}
            <div className="lg:col-span-2">
              <motion.div 
                className="glass-effect rounded-xl p-6 min-h-[600px]"
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.6 }}
              >
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center">
                    <TrendingUp className="h-6 w-6 text-cyan-400 mr-3" />
                    <h3 className="text-xl font-bold text-white">Visualization Display</h3>
                  </div>
                  
                  {currentVisualization && currentVisualization.success && (
                    <motion.button
                      onClick={handleExportVisualization}
                      className="text-sm bg-gray-700 hover:bg-gray-600 text-white px-3 py-2 rounded-lg transition-all duration-300"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <Download className="h-4 w-4 mr-2 inline" />
                      Export
                    </motion.button>
                  )}
                </div>

                {isCreatingVisualization ? (
                  <div className="flex items-center justify-center h-96">
                    <div className="text-center">
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                        className="mx-auto mb-4"
                      >
                        <Loader2 className="h-12 w-12 text-cyan-400" />
                      </motion.div>
                      <p className="text-gray-300">Creating visualization...</p>
                    </div>
                  </div>
                ) : currentVisualization?.success && currentVisualization?.html ? (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5 }}
                    className="visualization-container w-full h-full overflow-auto"
                    style={{ minHeight: '400px' }}
                  >
                    <iframe
                      srcDoc={currentVisualization.html}
                      className="w-full h-full border-0 rounded-lg"
                      style={{ minHeight: '500px', backgroundColor: 'transparent' }}
                      sandbox="allow-scripts allow-same-origin"
                      title="Visualization"
                    />
                  </motion.div>
                ) : currentVisualization && !currentVisualization.success ? (
                  <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4 flex items-center justify-center h-96">
                    <div className="text-center">
                      <AlertCircle className="h-12 w-12 text-red-400 mx-auto mb-4" />
                      <h4 className="text-lg font-medium text-red-400 mb-2">Visualization Error</h4>
                      <p className="text-red-300">{currentVisualization.error || 'Unknown error occurred'}</p>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-96 text-center">
                    <div>
                      <BarChart3 className="h-24 w-24 text-gray-600 mx-auto mb-6" />
                      <h4 className="text-xl font-medium text-gray-300 mb-2">
                        No Visualization Selected
                      </h4>
                      <p className="text-gray-400">
                        Choose a visualization type from the control panel to get started
                      </p>
                    </div>
                  </div>
                )}

                {currentVisualization && !currentVisualization.success && (
                  <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4 mt-4">
                    <div className="flex items-center text-red-400">
                      <AlertCircle className="h-5 w-5 mr-2" />
                      <span className="font-medium">Visualization Error</span>
                    </div>
                    <p className="text-red-300 mt-1">{currentVisualization.error}</p>
                  </div>
                )}
              </motion.div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}