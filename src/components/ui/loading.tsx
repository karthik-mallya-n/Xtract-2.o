/**
 * Loading Spinner Component - Shows loading state
 */
interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function LoadingSpinner({ size = 'md', className = '' }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-8 w-8'
  };
  
  return (
    <div className={`animate-spin rounded-full border-2 border-gray-300 border-t-blue-600 ${sizeClasses[size]} ${className}`}></div>
  );
}

/**
 * Progress Bar Component - Shows progress with percentage
 */
interface ProgressBarProps {
  progress: number;
  className?: string;
  showPercentage?: boolean;
  color?: 'blue' | 'green' | 'orange' | 'red';
}

export function ProgressBar({ progress, className = '', showPercentage = true, color = 'blue' }: ProgressBarProps) {
  const colorClasses = {
    blue: 'bg-blue-600',
    green: 'bg-green-600',
    orange: 'bg-orange-600',
    red: 'bg-red-600'
  };
  
  return (
    <div className={className}>
      {showPercentage && (
        <div className="flex justify-between text-sm font-medium text-gray-700 mb-1">
          <span>Progress</span>
          <span>{Math.round(progress)}%</span>
        </div>
      )}
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div 
          className={`h-2 rounded-full transition-all duration-300 ease-out ${colorClasses[color]}`}
          style={{ width: `${Math.min(progress, 100)}%` }}
        />
      </div>
    </div>
  );
}

/**
 * Alert Component - Shows important messages
 */
interface AlertProps {
  children: React.ReactNode;
  type?: 'info' | 'success' | 'warning' | 'error';
  className?: string;
}

export function Alert({ children, type = 'info', className = '' }: AlertProps) {
  const typeClasses = {
    info: 'bg-blue-50 border-blue-200 text-blue-800',
    success: 'bg-green-50 border-green-200 text-green-800',
    warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
    error: 'bg-red-50 border-red-200 text-red-800'
  };
  
  return (
    <div className={`border rounded-lg p-4 ${typeClasses[type]} ${className}`}>
      {children}
    </div>
  );
}