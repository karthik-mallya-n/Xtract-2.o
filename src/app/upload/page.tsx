'use client';

import { useState, useEffect } from 'react';
import { Upload, FileText, Loader2, AlertCircle } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';

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
    <div
      className={`relative border-2 border-dashed rounded-lg  text-center transition-colors duration-200 ${
        dragActive 
          ? 'border-blue-500 bg-blue-50' 
          : 'border-gray-300 hover:border-gray-400'
      } ${isLoading ? 'opacity-50 pointer-events-none' : ''}`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      <div className="flex flex-col items-center">
        {isLoading ? (
          <Loader2 className="h-12 w-12 text-blue-600 animate-spin mb-4" />
        ) : (
          <Upload className="h-12 w-12 text-gray-400 mb-4" />
        )}
        
        <h3 className="text-lg font-medium text-gray-900 mb-2">
          {isLoading ? 'Processing...' : 'Upload your dataset'}
        </h3>
        
        <p className="text-gray-600 mb-4">
          Drag and drop your file here, or click to browse
        </p>
        
        <input
          type="file"
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          onChange={handleFileInput}
          accept=".csv,.json,.xlsx,.xls"
          disabled={isLoading}
        />
        
        <button
          className="px-6 py-2 text-sm font-medium text-blue-600 bg-blue-50 hover:bg-blue-100 rounded-md transition-colors duration-200"
          disabled={isLoading}
        >
          Browse Files
        </button>
        
        <p className="text-xs text-gray-500 mt-2">
          Supports CSV, JSON, Excel files
        </p>
      </div>
    </div>
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
    <div className="min-h-screen bg-gray-50" style={{paddingTop: '120px', paddingBottom: '48px'}}>
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Backend Status Warning */}
        {backendAvailable === false && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-md p-4">
            <div className="flex">
              <AlertCircle className="h-5 w-5 text-red-400" />
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">
                  Backend Not Available
                </h3>
                <p className="mt-1 text-sm text-red-700">
                  The Flask server is not running. Please start it by running 
                  <code className="mx-1 px-1 bg-red-100 rounded">python app.py</code> 
                  in the my_flask_app directory.
                </p>
              </div>
            </div>
          </div>
        )}

        <div className="bg-white rounded-lg shadow-lg overflow-hidden">
          <div className="px-8 py-6 border-b border-gray-200">
            <h1 className="text-3xl font-bold text-gray-900">Upload Your Dataset</h1>
            <p className="text-lg text-gray-600 mt-2">
              Get started by uploading your data and telling us about its characteristics.
            </p>
          </div>
          
          <form onSubmit={handleSubmit} className="p-8 space-y-8">
            {/* Error Message */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-md p-4">
                <div className="flex">
                  <AlertCircle className="h-5 w-5 text-red-400" />
                  <div className="ml-3">
                    <p className="text-sm text-red-700">{error}</p>
                  </div>
                </div>
              </div>
            )}

            {/* File Upload Section */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-4">
                Select Your Dataset File
              </label>
              <FileUpload onFileSelect={handleFileSelect} isLoading={isLoading} />
              
              {selectedFile && (
                <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                  <div className="flex items-center">
                    <FileText className="h-5 w-5 text-blue-600 mr-2" />
                    <span className="text-sm font-medium text-blue-900">
                      {selectedFile.name}
                    </span>
                    <span className="text-sm text-blue-700 ml-2">
                      ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                    </span>
                  </div>
                </div>
              )}

              {/* Upload Progress */}
              {isLoading && uploadProgress > 0 && (
                <div className="mt-4">
                  <div className="flex justify-between text-sm text-gray-600 mb-1">
                    <span>Uploading...</span>
                    <span>{uploadProgress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${uploadProgress}%` }}
                    ></div>
                  </div>
                </div>
              )}
            </div>

            {/* Data Characteristics Section */}
            <div className="grid md:grid-cols-2 gap-6">
              {/* Data Labels Question */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Is your data labeled or unlabeled?
                </label>
                <div className="space-y-2">
                  <label className="flex items-center">
                    <input
                      type="radio"
                      name="isLabeled"
                      value="labeled"
                      checked={isLabeled === 'labeled'}
                      onChange={(e) => setIsLabeled(e.target.value)}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                    />
                    <span className="ml-2 text-sm text-gray-700">Labeled</span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="radio"
                      name="isLabeled"
                      value="unlabeled"
                      checked={isLabeled === 'unlabeled'}
                      onChange={(e) => setIsLabeled(e.target.value)}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                    />
                    <span className="ml-2 text-sm text-gray-700">Unlabeled</span>
                  </label>
                </div>
              </div>

              {/* Data Type Question */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Is your data continuous or categorical?
                </label>
                <div className="space-y-2">
                  <label className="flex items-center">
                    <input
                      type="radio"
                      name="dataType"
                      value="continuous"
                      checked={dataType === 'continuous'}
                      onChange={(e) => setDataType(e.target.value)}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                    />
                    <span className="ml-2 text-sm text-gray-700">Continuous</span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="radio"
                      name="dataType"
                      value="categorical"
                      checked={dataType === 'categorical'}
                      onChange={(e) => setDataType(e.target.value)}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                    />
                    <span className="ml-2 text-sm text-gray-700">Categorical</span>
                  </label>
                </div>
              </div>
            </div>

            {/* Submit Button */}
            <div className="flex justify-end">
              <button
                type="submit"
                disabled={!selectedFile || !isLabeled || !dataType || isLoading}
                className={`px-8 py-3 text-base font-medium rounded-lg transition-colors duration-200 ${
                  !selectedFile || !isLabeled || !dataType || isLoading
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-blue-600 text-white hover:bg-blue-700'
                }`}
              >
                {isLoading ? (
                  <span className="flex items-center">
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                    Processing...
                  </span>
                ) : (
                  'Process Dataset'
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}