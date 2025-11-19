'use client';

import { motion } from 'framer-motion';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  color?: 'cyan' | 'purple' | 'white';
}

export default function LoadingSpinner({ size = 'md', color = 'cyan' }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12'
  };

  const colorClasses = {
    cyan: 'border-cyan-400',
    purple: 'border-purple-400',
    white: 'border-white'
  };

  return (
    <motion.div
      className={`${sizeClasses[size]} border-2 border-transparent border-t-2 ${colorClasses[color]} rounded-full`}
      animate={{ rotate: 360 }}
      transition={{
        duration: 1,
        repeat: Infinity,
        ease: "linear"
      }}
      style={{
        filter: `drop-shadow(0 0 10px ${color === 'cyan' ? '#00f5ff' : color === 'purple' ? '#9945ff' : '#ffffff'})`
      }}
    />
  );
}