'use client';

import { useState } from 'react';
import { Upload, FileText, Loader2 } from 'lucide-react';
import Link from 'next/link';

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
      className={`relative border-2 border-dashed rounded-lg p-12 text-center transition-colors duration-200 ${
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
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isLabeled, setIsLabeled] = useState<string>('');
  const [dataType, setDataType] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadComplete, setUploadComplete] = useState(false);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!selectedFile || !isLabeled || !dataType) {
      alert('Please fill in all fields');
      return;
    }

    setIsLoading(true);
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    setIsLoading(false);
    setUploadComplete(true);
  };

  if (uploadComplete) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-8 text-center">
          <div className="flex justify-center mb-6">
            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center">
              <FileText className="h-8 w-8 text-green-600" />
            </div>
          </div>
          
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Upload Complete!
          </h2>
          
          <p className="text-gray-600 mb-6">
            Your dataset has been processed successfully. Our AI is now analyzing your data to recommend the best models.
          </p>
          
          <Link
            href="/select-model"
            className="inline-flex items-center px-6 py-3 text-base font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors duration-200"
          >
            View Model Recommendations
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="bg-white rounded-lg shadow-lg overflow-hidden">
          <div className="px-8 py-6 border-b border-gray-200">
            <h1 className="text-3xl font-bold text-gray-900">Upload Your Dataset</h1>
            <p className="text-lg text-gray-600 mt-2">
              Get started by uploading your data and telling us about its characteristics.
            </p>
          </div>
          
          <form onSubmit={handleSubmit} className="p-8 space-y-8">
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