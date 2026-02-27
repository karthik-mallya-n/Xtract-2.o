'use client';

import { useState, useEffect } from 'react';
import { Upload, FileText, Loader2, AlertCircle, Brain, Zap, Sparkles } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { api } from '@/lib/api';
import ParticleBackground from '@/components/ParticleBackground';

/**
 * File Upload Component - Handles drag and drop file upload
 * @param onFileSelect - Callback function when file is selected
 * @param isLoading - Loading state for the upload process
 */
interface FileUploadProps {
  onFileSelect: (file: File) => void;
  isLoading: boolean;
}

function FileUpload({ onFileSelect, isLoading }: FileUploadProps) {
  const [dragActive, setDragActive] = useState(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      onFileSelect(files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFileSelect(files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
  };

  return (
    <motion.div
      className={`futuristic-card relative border-2 border-dashed text-center transition-all duration-300 ${
        dragActive 
          ? 'border-cyan-400 bg-gray-800/30' 
          : 'border-gray-600 hover:border-cyan-400/50'
      } ${isLoading ? 'opacity-50 pointer-events-none' : ''}`}
      style={{ padding: '48px' }}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <div className="flex flex-col items-center">
        {isLoading ? (
          <Loader2 className="h-16 w-16 text-cyan-400 mb-6 animate-spin" />
        ) : (
          <motion.div
            className="p-6 rounded-full mb-6"
            style={{
              background: 'linear-gradient(135deg, rgba(0, 245, 255, 0.2) 0%, rgba(153, 69, 255, 0.2) 100%)',
              border: '1px solid rgba(0, 245, 255, 0.3)'
            }}
            whileHover={{ scale: 1.05, boxShadow: '0 0 20px rgba(0, 245, 255, 0.3)' }}
          >
            <Upload className="h-12 w-12 text-cyan-400" />
          </motion.div>
        )}
        
        <h3 className="text-2xl font-bold text-white mb-4">
          {isLoading ? 'Processing...' : 'Upload Your Dataset'}
        </h3>
        
        <p className="text-gray-300 mb-8 text-lg leading-relaxed">
          Drag and drop your file here, or click to browse
        </p>
        
        <input
          type="file"
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          onChange={handleFileInput}
          accept=".csv,.json,.xlsx,.xls"
          disabled={isLoading}
        />
        
        <motion.button
          className="px-8 py-4 text-lg font-bold rounded-xl"
          style={{
            background: 'linear-gradient(135deg, #00f5ff 0%, #9945ff 100%)',
            border: 'none',
            color: '#ffffff',
            fontWeight: '700',
            textTransform: 'uppercase',
            letterSpacing: '0.8px',
            boxShadow: '0 8px 32px rgba(0, 245, 255, 0.3)'
          }}
          disabled={isLoading}
          whileHover={{ 
            scale: 1.05,
            boxShadow: '0 0 30px rgba(0, 245, 255, 0.4)'
          }}
          whileTap={{ scale: 0.98 }}
        >
          <Zap className="h-5 w-5 mr-2 inline" />
          Browse Files
        </motion.button>
        
        <p className="text-cyan-400 text-sm mt-6 font-medium">
          Supports CSV, JSON, Excel files
        </p>
      </div>
    </motion.div>
  );
}

/**
 * Dataset Upload Page - Main page for uploading datasets and specifying data characteristics
 */
export default function UploadPage() {
  const router = useRouter();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isLabeled, setIsLabeled] = useState<string>('');
  const [dataType, setDataType] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string>('');
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(null);
  const [targetColumn, setTargetColumn] = useState<string>('');
  const [availableColumns, setAvailableColumns] = useState<string[]>([]);
  const [columnsPreviewed, setColumnsPreviewed] = useState(false);
  const [selectedTrainingColumns, setSelectedTrainingColumns] = useState<string[]>([]);
  const [showColumnSelector, setShowColumnSelector] = useState(false);

  // Check backend availability on component mount
  useEffect(() => {
    const checkBackend = async () => {
      const available = await api.isBackendAvailable();
      setBackendAvailable(available);
    };
    
    checkBackend();
  }, []);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setError('');
    setAvailableColumns([]);
    setTargetColumn('');
    setColumnsPreviewed(false);
    setSelectedTrainingColumns([]);
    setShowColumnSelector(false);
    // Preview columns from backend after file select
    const formData = new FormData();
    formData.append('file', file);
    fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'}/api/preview-columns`, {
      method: 'POST',
      body: formData
    })
      .then(res => res.json())
      .then(data => {
        if (data.success && Array.isArray(data.columns)) {
          setAvailableColumns(data.columns);
          setSelectedTrainingColumns(data.columns); // Initially select all columns
          setColumnsPreviewed(true);
          setShowColumnSelector(true);
        }
      })
      .catch(err => {
        console.error('Error previewing columns:', err);
        // Try to read columns from file directly as fallback
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const text = e.target?.result as string;
            const lines = text.split('\n');
            if (lines.length > 0) {
              const headers = lines[0].split(',').map(h => h.trim());
              setAvailableColumns(headers);
              setSelectedTrainingColumns(headers); // Initially select all columns
              setColumnsPreviewed(true);
              setShowColumnSelector(true);
            }
          } catch (err) {
            console.error('Error parsing file:', err);
          }
        };
        reader.readAsText(file);
      });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!selectedFile || !isLabeled) {
      setError('Please fill in all fields');
      return;
    }
    if (isLabeled === 'labeled') {
      if (!dataType) {
        setError('Please select whether your data is continuous or categorical.');
        return;
      }
      if (availableColumns.length > 0 && !targetColumn) {
        setError('Please select the target column.');
        return;
      }
      if (selectedTrainingColumns.length === 0) {
        setError('Please select at least one column for training.');
        return;
      }
      if (targetColumn && !selectedTrainingColumns.includes(targetColumn)) {
        setError('Target column must be included in training columns.');
        return;
      }
    }
    // For unlabeled data, clear dataType
    if (isLabeled === 'unlabeled') {
      setDataType('');
    }

    if (!backendAvailable) {
      setError('Backend server is not available. Please make sure the Flask server is running.');
      return;
    }

    setIsLoading(true);
    setError('');
    setUploadProgress(0);
    
    try {
      const response = await api.uploadWithProgress(
        selectedFile,
        isLabeled,
        dataType,
        targetColumn,
        (progress) => setUploadProgress(progress),
        selectedTrainingColumns
      );

      if (response.success && response.file_id) {
        localStorage.setItem('currentFileId', response.file_id);
        localStorage.setItem('currentFileName', response.filename || selectedFile.name);
        router.push('/select-model');
      } else if (response.available_columns) {
        // Backend requests target column selection
        setAvailableColumns(response.available_columns);
        setError(response.error || 'Please select the target column.');
      } else {
        setError(response.error || 'Upload failed');
      }
    } catch (err) {
      console.error('Upload error:', err);
      setError('Upload failed. Please check your connection and try again.');
    } finally {
      setIsLoading(false);
      setUploadProgress(0);
    }
  };

  return (
    <div className="min-h-screen relative overflow-hidden bg-gray-900">
      {/* Background Elements */}
      <ParticleBackground />
      <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-purple-900 to-cyan-900 opacity-20" />
      <div className="absolute inset-0 geometric-pattern opacity-30" />
      
      {/* Main Container */}
      <div className="relative z-10 min-h-screen" style={{paddingTop: '50px', paddingBottom: '48px'}}>
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Header Section */}
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
                className="relative"
                animate={{ 
                  scale: [1, 1.1, 1]
                }}
                transition={{ 
                  scale: { duration: 2, repeat: Infinity, ease: "easeInOut" }
                }}
                style={{
                  filter: 'drop-shadow(0 0 20px rgba(0, 245, 255, 0.5))'
                }}
              >
                <Brain className="h-20 w-20 text-cyan-400" />
              </motion.div>
            </motion.div>
            
            <h1 className="text-5xl sm:text-6xl font-black leading-tight tracking-tight mb-6">
              <span className="block text-white mb-2">Upload Your</span>
              <span 
                className="block"
                style={{
                  background: 'linear-gradient(135deg, #00f5ff 0%, #9945ff 100%)',
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                Dataset
              </span>
            </h1>
            <p className="text-xl text-gray-300 leading-relaxed max-w-3xl mx-auto">
              Get started by uploading your data and telling us about its characteristics.
            </p>
          </motion.div>

          {/* Backend Status Warning */}
          {backendAvailable === false && (
            <motion.div 
              className="mb-8 futuristic-card border-red-500/30 bg-red-900/20"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <div className="flex items-start">
                <AlertCircle className="h-6 w-6 text-red-400 mr-4 mt-1" />
                <div>
                  <h3 className="text-lg font-bold text-red-400 mb-2">
                    Backend Not Available
                  </h3>
                  <p className="text-red-300">
                    The Flask server is not running. Please start it by running 
                    <code className="mx-1 px-2 py-1 bg-red-800/50 rounded text-red-200 font-mono">python app.py</code> 
                    in the my_flask_app directory.
                  </p>
                </div>
              </div>
            </motion.div>
          )}

          <motion.div 
            className="futuristic-card"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
          >
            <div className="mb-8">
              <div className="flex items-center mb-4">
                <Sparkles className="h-8 w-8 text-cyan-400 mr-3" />
                <h2 className="text-3xl font-bold text-white">Dataset Upload</h2>
              </div>
              <p className="text-gray-300 text-lg">
                Upload your dataset and configure the training parameters.
              </p>
            </div>
          
          <form onSubmit={handleSubmit} className="space-y-8">
            {/* Error Message */}
            {error && (
              <motion.div 
                className="futuristic-card border-red-500/30 bg-red-900/20"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
              >
                <div className="flex items-start">
                  <AlertCircle className="h-6 w-6 text-red-400 mr-4 mt-1" />
                  <div>
                    <p className="text-red-300 text-lg">{error}</p>
                  </div>
                </div>
              </motion.div>
            )}

            {/* File Upload Section */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              <label className="block text-lg font-bold text-white mb-6">
                Select Your Dataset File
              </label>
              <FileUpload onFileSelect={handleFileSelect} isLoading={isLoading} />
              
              {selectedFile && (
                <motion.div 
                  className="mt-6 futuristic-card bg-cyan-900/20 border-cyan-400/30"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="flex items-center">
                    <div className="p-2 rounded-lg bg-cyan-400/20 mr-4">
                      <FileText className="h-6 w-6 text-cyan-400" />
                    </div>
                    <div>
                      <span className="text-lg font-bold text-white block">
                        {selectedFile.name}
                      </span>
                      <span className="text-cyan-300">
                        ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                      </span>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Upload Progress */}
              {isLoading && uploadProgress > 0 && (
                <motion.div 
                  className="mt-6"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="flex justify-between text-lg text-cyan-300 mb-3">
                    <span className="font-medium">Uploading...</span>
                    <span className="font-bold">{uploadProgress}%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                    <motion.div 
                      className="h-3 rounded-full"
                      style={{ 
                        background: 'linear-gradient(135deg, #00f5ff 0%, #9945ff 100%)',
                        boxShadow: '0 0 10px rgba(0, 245, 255, 0.5)'
                      }}
                      initial={{ width: '0%' }}
                      animate={{ width: `${uploadProgress}%` }}
                      transition={{ duration: 0.5, ease: 'easeOut' }}
                    />
                  </div>
                </motion.div>
              )}
            </motion.div>

            {/* Data Characteristics Section */}
            <motion.div 
              className={`grid gap-8 ${isLabeled === 'labeled' ? 'md:grid-cols-2' : 'md:grid-cols-1'}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.5 }}
            >
              {/* Data Labels Question */}
              <div className="futuristic-card">
                <label className="block text-xl font-bold text-white mb-6">
                  Is your data labeled or unlabeled?
                </label>
                <div className="space-y-4">
                  <label className="flex items-center p-4 rounded-xl bg-gray-800/30 hover:bg-gray-700/30 transition-all duration-200 cursor-pointer border border-gray-600 hover:border-cyan-400/50">
                    <input
                      type="radio"
                      name="isLabeled"
                      value="labeled"
                      checked={isLabeled === 'labeled'}
                      onChange={(e) => setIsLabeled(e.target.value)}
                      className="h-5 w-5 text-cyan-400 focus:ring-cyan-400 focus:ring-2 bg-gray-700 border-gray-500"
                    />
                    <span className="ml-4 text-lg text-white font-medium">Labeled</span>
                  </label>
                  <label className="flex items-center p-4 rounded-xl bg-gray-800/30 hover:bg-gray-700/30 transition-all duration-200 cursor-pointer border border-gray-600 hover:border-cyan-400/50">
                    <input
                      type="radio"
                      name="isLabeled"
                      value="unlabeled"
                      checked={isLabeled === 'unlabeled'}
                      onChange={(e) => {
                        setIsLabeled(e.target.value);
                        setDataType(''); // Clear data type when switching to unlabeled
                      }}
                      className="h-5 w-5 text-cyan-400 focus:ring-cyan-400 focus:ring-2 bg-gray-700 border-gray-500"
                    />
                    <span className="ml-4 text-lg text-white font-medium">Unlabeled</span>
                  </label>
                </div>
              </div>

              {/* Data Type Question - Only show for labeled data */}
              {isLabeled === 'labeled' && (
                <div className="futuristic-card">
                  <label className="block text-xl font-bold text-white mb-6">
                    Is your data continuous or categorical?
                  </label>
                  <div className="space-y-4">
                    <label className="flex items-center p-4 rounded-xl bg-gray-800/30 hover:bg-gray-700/30 transition-all duration-200 cursor-pointer border border-gray-600 hover:border-cyan-400/50">
                      <input
                        type="radio"
                        name="dataType"
                        value="continuous"
                        checked={dataType === 'continuous'}
                        onChange={(e) => setDataType(e.target.value)}
                        className="h-5 w-5 text-cyan-400 focus:ring-cyan-400 focus:ring-2 bg-gray-700 border-gray-500"
                      />
                      <span className="ml-4 text-lg text-white font-medium">Continuous</span>
                    </label>
                    <label className="flex items-center p-4 rounded-xl bg-gray-800/30 hover:bg-gray-700/30 transition-all duration-200 cursor-pointer border border-gray-600 hover:border-cyan-400/50">
                      <input
                        type="radio"
                        name="dataType"
                        value="categorical"
                        checked={dataType === 'categorical'}
                        onChange={(e) => setDataType(e.target.value)}
                        className="h-5 w-5 text-cyan-400 focus:ring-cyan-400 focus:ring-2 bg-gray-700 border-gray-500"
                      />
                      <span className="ml-4 text-lg text-white font-medium">Categorical</span>
                    </label>
                  </div>
                </div>
              )}
            </motion.div>

            {/* Target Column Selection */}
            {isLabeled === 'labeled' && availableColumns.length > 0 && (
              <motion.div 
                className="futuristic-card mt-8"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
              >
                <label className="block text-xl font-bold text-white mb-4">
                  Select Target Attribute
                </label>
                <select
                  className="w-full px-4 py-3 rounded-lg bg-gray-800 text-white text-lg border border-cyan-400/30 focus:outline-none focus:ring-2 focus:ring-cyan-400"
                  value={targetColumn}
                  onChange={e => setTargetColumn(e.target.value)}
                >
                  <option value="">-- Select Target Column --</option>
                  {availableColumns.map(col => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </motion.div>
            )}

            {/* Training Columns Selection */}
            {showColumnSelector && availableColumns.length > 0 && (
              <motion.div 
                className="futuristic-card mt-8"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
              >
                <label className="block text-xl font-bold text-white mb-4">
                  Select Columns for Training
                </label>
                <p className="text-gray-400 text-sm mb-6">
                  Choose which columns to include in the model training. You can exclude irrelevant columns like IDs or timestamps.
                </p>
                
                {/* Select All / Deselect All Buttons */}
                <div className="flex gap-4 mb-6">
                  <motion.button
                    type="button"
                    onClick={() => setSelectedTrainingColumns(availableColumns)}
                    className="px-4 py-2 text-sm font-medium text-cyan-400 bg-cyan-400/10 border border-cyan-400/30 rounded-lg hover:bg-cyan-400/20 transition-all duration-200"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    Select All
                  </motion.button>
                  <motion.button
                    type="button"
                    onClick={() => setSelectedTrainingColumns([])}
                    className="px-4 py-2 text-sm font-medium text-red-400 bg-red-400/10 border border-red-400/30 rounded-lg hover:bg-red-400/20 transition-all duration-200"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    Deselect All
                  </motion.button>
                </div>

                {/* Column Checkboxes */}
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {availableColumns.map((column, index) => (
                    <motion.label
                      key={column}
                      className={`flex items-center p-3 rounded-lg cursor-pointer transition-all duration-200 border ${
                        selectedTrainingColumns.includes(column)
                          ? 'bg-cyan-400/10 border-cyan-400/30 text-white'
                          : 'bg-gray-800/30 border-gray-600 hover:bg-gray-700/30 hover:border-cyan-400/20 text-gray-300'
                      }`}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3, delay: index * 0.05 }}
                      whileHover={{ scale: 1.02 }}
                    >
                      <input
                        type="checkbox"
                        checked={selectedTrainingColumns.includes(column)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedTrainingColumns([...selectedTrainingColumns, column]);
                          } else {
                            setSelectedTrainingColumns(selectedTrainingColumns.filter(col => col !== column));
                          }
                        }}
                        className="h-5 w-5 text-cyan-400 bg-gray-700 border-gray-500 rounded focus:ring-cyan-400 focus:ring-2"
                      />
                      <span className="ml-3 text-sm font-medium">
                        {column}
                        {column === targetColumn && (
                          <span className="ml-2 px-2 py-1 text-xs font-bold bg-purple-500/20 text-purple-300 rounded-full border border-purple-500/30">
                            TARGET
                          </span>
                        )}
                      </span>
                    </motion.label>
                  ))}
                </div>
                
                {selectedTrainingColumns.length > 0 && (
                  <div className="mt-4 p-3 bg-gray-800/50 rounded-lg border border-gray-600">
                    <p className="text-sm text-gray-300">
                      <span className="font-medium text-white">Selected:</span> {selectedTrainingColumns.length} of {availableColumns.length} columns
                      {targetColumn && !selectedTrainingColumns.includes(targetColumn) && (
                        <span className="block mt-1 text-orange-400 font-medium">
                          ⚠️ Target column must be included in training columns
                        </span>
                      )}
                    </p>
                  </div>
                )}
              </motion.div>
            )}

            {/* Submit Button */}
            <motion.div 
              className="flex justify-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
            >
              <motion.button
                type="submit"
                disabled={!selectedFile || !isLabeled || (isLabeled === 'labeled' && !dataType) || isLoading || (isLabeled === 'labeled' && availableColumns.length > 0 && !targetColumn) || (showColumnSelector && selectedTrainingColumns.length === 0)}
                className={`px-12 py-5 text-xl font-bold rounded-xl transition-all duration-300 ${
                  !selectedFile || !isLabeled || (isLabeled === 'labeled' && !dataType) || isLoading || (isLabeled === 'labeled' && availableColumns.length > 0 && !targetColumn) || (showColumnSelector && selectedTrainingColumns.length === 0)
                    ? 'bg-gray-600 text-gray-400 cursor-not-allowed border border-gray-500'
                    : 'text-white border border-cyan-400/30'
                }`}
                style={{
                  background: !selectedFile || !isLabeled || (isLabeled === 'labeled' && !dataType) || isLoading || (isLabeled === 'labeled' && availableColumns.length > 0 && !targetColumn) || (showColumnSelector && selectedTrainingColumns.length === 0)
                    ? 'rgba(75, 85, 99, 0.5)' 
                    : 'linear-gradient(135deg, #00f5ff 0%, #9945ff 100%)',
                  boxShadow: !selectedFile || !isLabeled || (isLabeled === 'labeled' && !dataType) || isLoading || (isLabeled === 'labeled' && availableColumns.length > 0 && !targetColumn) || (showColumnSelector && selectedTrainingColumns.length === 0)
                    ? 'none'
                    : '0 8px 32px rgba(0, 245, 255, 0.3)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.8px'
                }}
                whileHover={!selectedFile || !isLabeled || (isLabeled === 'labeled' && !dataType) || isLoading || (isLabeled === 'labeled' && availableColumns.length > 0 && !targetColumn) || (showColumnSelector && selectedTrainingColumns.length === 0) ? {} : { 
                  scale: 1.05,
                  boxShadow: '0 0 30px rgba(0, 245, 255, 0.4)'
                }}
                whileTap={!selectedFile || !isLabeled || (isLabeled === 'labeled' && !dataType) || isLoading || (isLabeled === 'labeled' && availableColumns.length > 0 && !targetColumn) || (showColumnSelector && selectedTrainingColumns.length === 0) ? {} : { scale: 0.98 }}
              >
                {isLoading ? (
                  <span className="flex items-center">
                    <Loader2 className="h-6 w-6 mr-3 animate-spin" />
                    Processing...
                  </span>
                ) : (
                  <span className="flex items-center">
                    <Brain className="h-6 w-6 mr-3" />
                    Process Dataset
                  </span>
                )}
              </motion.button>
            </motion.div>
          </form>
          </motion.div>
        </div>
      </div>
    </div>
  );
}