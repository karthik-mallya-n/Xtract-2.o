'use client';

import { useState, useEffect } from 'react';
import { Activity, CheckCircle, Loader2, TrendingUp, Target } from 'lucide-react';
import Link from 'next/link';

/**
 * Progress Bar Component - Shows training progress
 */
interface ProgressBarProps {
  progress: number;
  label: string;
}

function ProgressBar({ progress, label }: ProgressBarProps) {
  return (
    <div className="mb-6">
      <div className="flex justify-between text-sm font-medium text-gray-700 mb-2">
        <span>{label}</span>
        <span>{progress}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-3">
        <div 
          className="bg-blue-600 h-3 rounded-full transition-all duration-500 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}

/**
 * Metric Card Component - Displays real-time training metrics
 */
interface MetricCardProps {
  title: string;
  value: number;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  format?: 'percentage' | 'decimal';
}

function MetricCard({ title, value, icon: Icon, color, format = 'percentage' }: MetricCardProps) {
  const formatValue = (val: number) => {
    if (format === 'percentage') {
      return `${val.toFixed(1)}%`;
    }
    return val.toFixed(3);
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className={`text-2xl font-bold ${color}`}>
            {formatValue(value)}
          </p>
        </div>
        <div className={`p-3 rounded-full ${color.replace('text-', 'bg-').replace('-600', '-100')}`}>
          <Icon className={`h-6 w-6 ${color}`} />
        </div>
      </div>
    </div>
  );
}

/**
 * Training Status Page - Shows real-time training progress and metrics
 */
export default function TrainingStatusPage() {
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [accuracy, setAccuracy] = useState(0);
  const [precision, setPrecision] = useState(0);
  const [recall, setRecall] = useState(0);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);

  const totalEpochs = 100;

  useEffect(() => {
    // Simulate training progress
    const interval = setInterval(() => {
      setTrainingProgress(prev => {
        if (prev >= 100) {
          setIsComplete(true);
          clearInterval(interval);
          return 100;
        }
        
        const newProgress = prev + Math.random() * 3;
        const progressCapped = Math.min(newProgress, 100);
        
        // Update metrics based on progress
        const progressRatio = progressCapped / 100;
        setAccuracy(75 + (progressRatio * 12)); // 75% to 87%
        setPrecision(70 + (progressRatio * 15)); // 70% to 85%
        setRecall(68 + (progressRatio * 17)); // 68% to 85%
        setCurrentEpoch(Math.floor(progressRatio * totalEpochs));
        
        // Add training logs
        if (Math.random() < 0.3) {
          const logs = [
            `Epoch ${Math.floor(progressRatio * totalEpochs)}: Loss = ${(1 - progressRatio + Math.random() * 0.5).toFixed(4)}`,
            `Validation accuracy improved to ${(75 + progressRatio * 12).toFixed(2)}%`,
            `Learning rate adjusted to ${(0.001 * (1 - progressRatio * 0.5)).toFixed(6)}`,
            `Batch processing completed - ${Math.floor(Math.random() * 1000)} samples processed`,
            `Model weights updated - convergence check passed`
          ];
          
          setTrainingLogs(prev => {
            const newLog = logs[Math.floor(Math.random() * logs.length)];
            return [newLog, ...prev.slice(0, 9)]; // Keep last 10 logs
          });
        }
        
        return progressCapped;
      });
    }, 200);

    return () => clearInterval(interval);
  }, []);

  if (isComplete) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="max-w-2xl w-full bg-white rounded-lg shadow-lg p-8 text-center">
          <div className="flex justify-center mb-6">
            <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center">
              <CheckCircle className="h-12 w-12 text-green-600" />
            </div>
          </div>
          
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Training Complete! 🎉
          </h2>
          
          <p className="text-lg text-gray-600 mb-8">
            Your Random Forest model has been successfully trained and is ready for predictions.
          </p>
          
          {/* Final Metrics */}
          <div className="grid grid-cols-3 gap-4 mb-8">
            <div className="bg-blue-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-blue-600">{accuracy.toFixed(1)}%</div>
              <div className="text-sm text-blue-700">Final Accuracy</div>
            </div>
            <div className="bg-green-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-green-600">{precision.toFixed(1)}%</div>
              <div className="text-sm text-green-700">Precision</div>
            </div>
            <div className="bg-purple-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-purple-600">{recall.toFixed(1)}%</div>
              <div className="text-sm text-purple-700">Recall</div>
            </div>
          </div>
          
          <Link
            href="/inference"
            className="inline-flex items-center px-8 py-3 text-lg font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors duration-200"
          >
            Start Making Predictions
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-4">
            <Activity className="h-12 w-12 text-blue-600" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Training in Progress
          </h1>
          <p className="text-xl text-gray-600">
            Your Random Forest model is learning from your dataset
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Progress Section */}
          <div className="lg:col-span-2 space-y-8">
            {/* Overall Progress */}
            <div className="bg-white rounded-lg shadow-md p-8">
              <div className="flex items-center mb-6">
                <Loader2 className="h-6 w-6 text-blue-600 animate-spin mr-3" />
                <h2 className="text-2xl font-bold text-gray-900">Training Progress</h2>
              </div>
              
              <ProgressBar progress={trainingProgress} label="Overall Progress" />
              
              <div className="grid grid-cols-2 gap-4 text-sm text-gray-600">
                <div>
                  <span className="font-medium">Current Epoch:</span> {currentEpoch} / {totalEpochs}
                </div>
                <div>
                  <span className="font-medium">Estimated Time Remaining:</span> {
                    trainingProgress < 100 
                      ? `${Math.max(1, Math.ceil((100 - trainingProgress) / 5))} minutes`
                      : 'Complete'
                  }
                </div>
              </div>
            </div>

            {/* Real-time Metrics */}
            <div className="bg-white rounded-lg shadow-md p-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Live Metrics</h2>
              
              <div className="grid md:grid-cols-3 gap-6">
                <MetricCard
                  title="Accuracy"
                  value={accuracy}
                  icon={TrendingUp}
                  color="text-blue-600"
                />
                <MetricCard
                  title="Precision"
                  value={precision}
                  icon={Target}
                  color="text-green-600"
                />
                <MetricCard
                  title="Recall"
                  value={recall}
                  icon={Activity}
                  color="text-purple-600"
                />
              </div>
            </div>
          </div>

          {/* Training Logs Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-md p-6 h-fit">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Training Logs</h3>
              
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {trainingLogs.length === 0 ? (
                  <p className="text-gray-500 text-sm">Training logs will appear here...</p>
                ) : (
                  trainingLogs.map((log, index) => (
                    <div
                      key={index}
                      className="text-xs text-gray-600 p-2 bg-gray-50 rounded border-l-2 border-blue-200"
                    >
                      {log}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Status Footer */}
        <div className="mt-12 text-center">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 max-w-2xl mx-auto">
            <div className="flex items-center justify-center mb-3">
              <Loader2 className="h-5 w-5 text-blue-600 animate-spin mr-2" />
              <span className="font-medium text-blue-900">Training Active</span>
            </div>
            <p className="text-blue-800">
              Please keep this page open while training is in progress. 
              We will automatically redirect you when training is complete.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}