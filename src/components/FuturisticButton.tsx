'use client';

import { motion } from 'framer-motion';
import { ReactNode } from 'react';

interface FuturisticButtonProps {
  children: ReactNode;
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  onClick?: () => void;
  disabled?: boolean;
  loading?: boolean;
  className?: string;
}

export default function FuturisticButton({
  children,
  variant = 'primary',
  size = 'md',
  onClick,
  disabled = false,
  loading = false,
  className = ''
}: FuturisticButtonProps) {
  const sizeClasses = {
    sm: 'px-4 py-2 text-sm',
    md: 'px-6 py-3 text-base',
    lg: 'px-8 py-4 text-lg'
  };

  const variantStyles = {
    primary: {
      background: 'linear-gradient(135deg, #00f5ff 0%, #9945ff 100%)',
      color: '#ffffff',
      border: 'none'
    },
    secondary: {
      background: 'rgba(255, 255, 255, 0.05)',
      color: '#00f5ff',
      border: '1px solid rgba(0, 245, 255, 0.3)'
    },
    ghost: {
      background: 'transparent',
      color: '#a0a0b3',
      border: '1px solid rgba(160, 160, 179, 0.3)'
    }
  };

  return (
    <motion.button
      className={`
        ${sizeClasses[size]} 
        font-semibold rounded-lg transition-all duration-300 
        ${disabled || loading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
        ${className}
      `}
      style={variantStyles[variant]}
      whileHover={!disabled && !loading ? {
        scale: 1.05,
        boxShadow: variant === 'primary' 
          ? '0 0 30px rgba(0, 245, 255, 0.4)'
          : '0 0 20px rgba(0, 245, 255, 0.2)',
        transition: { duration: 0.3 }
      } : {}}
      whileTap={!disabled && !loading ? { scale: 0.98 } : {}}
      onClick={disabled || loading ? undefined : onClick}
      disabled={disabled || loading}
    >
      {loading ? (
        <span className="flex items-center justify-center">
          <motion.div
            className="w-4 h-4 border-2 border-transparent border-t-current rounded-full mr-2"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          />
          Loading...
        </span>
      ) : (
        children
      )}
      
      {/* Shine effect on hover */}
      <motion.div
        className="absolute inset-0 rounded-lg overflow-hidden pointer-events-none"
        style={{
          background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent)',
          transform: 'translateX(-100%)'
        }}
        whileHover={{
          transform: 'translateX(100%)',
          transition: { duration: 0.6 }
        }}
      />
    </motion.button>
  );
}