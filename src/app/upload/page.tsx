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
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          >
            <Loader2 className="h-16 w-16 text-cyan-400 mb-6" />
          </motion.div>
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
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!selectedFile || !isLabeled || !dataType) {
      setError('Please fill in all fields');
      return;
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
        (progress) => setUploadProgress(progress)
      );

      if (response.success && response.file_id) {
        // Store file_id in localStorage for use in other pages
        localStorage.setItem('currentFileId', response.file_id);
        localStorage.setItem('currentFileName', response.filename || selectedFile.name);
        
        // Navigate to model selection page
        router.push('/select-model');
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
                  rotateY: 360,
                  scale: [1, 1.1, 1]
                }}
                transition={{ 
                  rotateY: { duration: 8, repeat: Infinity, ease: "linear" },
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
              className="grid md:grid-cols-2 gap-8"
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
                      onChange={(e) => setIsLabeled(e.target.value)}
                      className="h-5 w-5 text-cyan-400 focus:ring-cyan-400 focus:ring-2 bg-gray-700 border-gray-500"
                    />
                    <span className="ml-4 text-lg text-white font-medium">Unlabeled</span>
                  </label>
                </div>
              </div>

              {/* Data Type Question */}
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
            </motion.div>

            {/* Submit Button */}
            <motion.div 
              className="flex justify-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
            >
              <motion.button
                type="submit"
                disabled={!selectedFile || !isLabeled || !dataType || isLoading}
                className={`px-12 py-5 text-xl font-bold rounded-xl transition-all duration-300 ${
                  !selectedFile || !isLabeled || !dataType || isLoading
                    ? 'bg-gray-600 text-gray-400 cursor-not-allowed border border-gray-500'
                    : 'text-white border border-cyan-400/30'
                }`}
                style={{
                  background: !selectedFile || !isLabeled || !dataType || isLoading 
                    ? 'rgba(75, 85, 99, 0.5)' 
                    : 'linear-gradient(135deg, #00f5ff 0%, #9945ff 100%)',
                  boxShadow: !selectedFile || !isLabeled || !dataType || isLoading
                    ? 'none'
                    : '0 8px 32px rgba(0, 245, 255, 0.3)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.8px'
                }}
                whileHover={!selectedFile || !isLabeled || !dataType || isLoading ? {} : { 
                  scale: 1.05,
                  boxShadow: '0 0 30px rgba(0, 245, 255, 0.4)'
                }}
                whileTap={!selectedFile || !isLabeled || !dataType || isLoading ? {} : { scale: 0.98 }}
              >
                {isLoading ? (
                  <span className="flex items-center">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    >
                      <Loader2 className="h-6 w-6 mr-3" />
                    </motion.div>
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