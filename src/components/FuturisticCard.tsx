'use client';

import { motion } from 'framer-motion';
import { ReactNode } from 'react';

interface FuturisticCardProps {
  children: ReactNode;
  className?: string;
  hover?: boolean;
  glow?: boolean;
  onClick?: () => void;
}

export default function FuturisticCard({ 
  children, 
  className = '', 
  hover = true, 
  glow = false,
  onClick 
}: FuturisticCardProps) {
  return (
    <motion.div
      className={`futuristic-card ${className} ${onClick ? 'cursor-pointer' : ''}`}
      style={{
        background: 'rgba(255, 255, 255, 0.05)',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: '16px',
        boxShadow: glow ? '0 0 20px rgba(0, 245, 255, 0.2)' : undefined
      }}
      whileHover={hover ? {
        y: -5,
        scale: 1.02,
        borderColor: 'rgba(0, 245, 255, 0.3)',
        boxShadow: '0 0 30px rgba(0, 245, 255, 0.3)',
        transition: { duration: 0.3 }
      } : {}}
      whileTap={onClick ? { scale: 0.98 } : {}}
      onClick={onClick}
    >
      <div className="relative z-10">
        {children}
      </div>
      
      {/* Holographic effect overlay */}
      <motion.div
        className="absolute inset-0 rounded-2xl opacity-0 pointer-events-none"
        style={{
          background: 'linear-gradient(135deg, rgba(0, 245, 255, 0.1) 0%, rgba(153, 69, 255, 0.1) 100%)'
        }}
        whileHover={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      />
    </motion.div>
  );
}