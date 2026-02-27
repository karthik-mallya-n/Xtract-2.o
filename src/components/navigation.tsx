'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Database, Brain, Target, BarChart3, TrendingUp, MessageSquareText, Menu, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';

/**
 * Futuristic Navigation component with holographic effects and animations
 */
export default function Navigation() {
  const pathname = usePathname();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const navItems = [
    { href: '/', label: 'Home', icon: Brain },
    { href: '/upload', label: 'Upload Data', icon: Database },
    { href: '/select-model', label: 'Models', icon: Target },
    { href: '/results', label: 'Results', icon: BarChart3 },
    { href: '/visualizations', label: 'Visualizations', icon: TrendingUp },
    { href: '/analysis', label: 'Analysis', icon: MessageSquareText },
  ];

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  return (
    <motion.nav 
      className="fixed top-0 left-0 right-0 z-50 glass-effect border-b border-cyan-500/20"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      style={{
        background: 'rgba(10, 10, 15, 0.95)',
        backdropFilter: 'blur(20px)',
        height: '100px', // Increased from 72px
      }}
    >
      <div className="max-w-7xl mx-auto h-full" style={{paddingLeft: '48px', paddingRight: '48px'}}>
        <div className="flex justify-between items-center h-full">
          {/* Logo/Brand */}
          <motion.div 
            className="flex items-center"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
              <Link href="/" className="flex items-center space-x-4">
              <motion.div
                className="relative"
                animate={{ 
                  boxShadow: [
                    '0 0 20px rgba(0, 245, 255, 0.3)',
                    '0 0 30px rgba(153, 69, 255, 0.4)',
                    '0 0 20px rgba(0, 245, 255, 0.3)'
                  ]
                }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              >
                <Brain className="h-8 w-8 text-cyan-400" />
              </motion.div>
              <motion.span 
                className="text-xl font-bold gradient-text"
                style={{
                  background: 'linear-gradient(135deg, #00f5ff 0%, #9945ff 100%)',
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
                whileHover={{ 
                  scale: 1.1,
                  transition: { type: "spring", stiffness: 400 }
                }}
              >
                Xtract AI
              </motion.span>
            </Link>
          </motion.div>

          {/* Desktop Navigation Links */}
          <div className="hidden md:flex items-center space-x-3">
            {navItems.map((item, index) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              
              return (
                <motion.div
                  key={item.href}
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ 
                    duration: 0.6, 
                    delay: index * 0.1,
                    ease: "easeOut" 
                  }}
                >
                  <Link href={item.href}>
                    <motion.div
                      className={`relative flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
                        isActive
                          ? 'text-cyan-400'
                          : 'text-gray-300 hover:text-cyan-400'
                      }`}
                      whileHover={{ 
                        scale: 1.05,
                        y: -2,
                      }}
                      whileTap={{ scale: 0.95 }}
                      style={{
                        background: isActive 
                          ? 'rgba(0, 245, 255, 0.1)' 
                          : 'transparent'
                      }}
                    >
                      {/* Active indicator */}
                      {isActive && (
                        <motion.div
                          className="absolute inset-0 rounded-lg border border-cyan-500/30"
                          style={{
                            background: 'linear-gradient(135deg, rgba(0, 245, 255, 0.1) 0%, rgba(153, 69, 255, 0.1) 100%)',
                            boxShadow: '0 0 20px rgba(0, 245, 255, 0.2)'
                          }}
                          layoutId="activeTab"
                          transition={{ type: "spring", stiffness: 500, damping: 30 }}
                        />
                      )}
                      
                      {/* Icon with glow effect */}
                      <motion.div
                        className="relative z-10"
                        animate={isActive ? {
                          textShadow: '0 0 10px currentColor'
                        } : {}}
                      >
                        <Icon className="h-4 w-4" />
                      </motion.div>
                      
                      {/* Label */}
                      <span className="relative z-10">{item.label}</span>
                      
                      {/* Hover glow effect */}
                      <motion.div
                        className="absolute inset-0 rounded-lg opacity-0"
                        style={{
                          background: 'linear-gradient(135deg, rgba(0, 245, 255, 0.05) 0%, rgba(153, 69, 255, 0.05) 100%)',
                        }}
                        whileHover={{ opacity: 1 }}
                        transition={{ duration: 0.3 }}
                      />
                    </motion.div>
                  </Link>
                </motion.div>
              );
            })}
          </div>

          {/* Mobile menu button */}
          <motion.button
            className="md:hidden text-gray-300 hover:text-cyan-400"
            style={{padding: '12px'}}
            onClick={toggleMobileMenu}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <AnimatePresence mode="wait">
              {isMobileMenuOpen ? (
                <motion.div
                  key="close"
                  initial={{ rotate: -90, opacity: 0 }}
                  animate={{ rotate: 0, opacity: 1 }}
                  exit={{ rotate: 90, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <X className="h-6 w-6" />
                </motion.div>
              ) : (
                <motion.div
                  key="menu"
                  initial={{ rotate: 90, opacity: 0 }}
                  animate={{ rotate: 0, opacity: 1 }}
                  exit={{ rotate: -90, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <Menu className="h-6 w-6" />
                </motion.div>
              )}
            </AnimatePresence>
          </motion.button>
        </div>
      </div>

      {/* Mobile navigation menu */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <motion.div
            className="md:hidden border-t border-cyan-500/20"
            style={{
              background: 'rgba(17, 17, 24, 0.95)',
              backdropFilter: 'blur(20px)',
            }}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
          >
            <div style={{paddingLeft: '16px', paddingRight: '16px', paddingTop: '16px', paddingBottom: '16px'}} className="space-y-2">
              {navItems.map((item, index) => {
                const Icon = item.icon;
                const isActive = pathname === item.href;
                
                return (
                  <motion.div
                    key={item.href}
                    initial={{ x: -50, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ 
                      duration: 0.4,
                      delay: index * 0.1,
                      ease: "easeOut"
                    }}
                  >
                    <Link href={item.href} onClick={() => setIsMobileMenuOpen(false)}>
                      <motion.div
                        className={`flex items-center space-x-3 rounded-lg text-sm font-medium transition-all duration-300 ${
                          isActive
                            ? 'text-cyan-400 bg-cyan-500/10'
                            : 'text-gray-300 hover:text-cyan-400 hover:bg-gray-800/50'
                        }`}
                        style={{
                          paddingLeft: '12px',
                          paddingRight: '12px',
                          paddingTop: '12px', 
                          paddingBottom: '12px'
                        }}
                        whileHover={{ x: 10, scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <motion.div
                          animate={isActive ? {
                            textShadow: '0 0 10px currentColor'
                          } : {}}
                        >
                          <Icon className="h-5 w-5" />
                        </motion.div>
                        <span>{item.label}</span>
                        
                        {/* Active indicator */}
                        {isActive && (
                          <motion.div
                            className="w-2 h-2 rounded-full bg-cyan-400"
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            style={{ boxShadow: '0 0 10px rgba(0, 245, 255, 0.5)' }}
                          />
                        )}
                      </motion.div>
                    </Link>
                  </motion.div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  );
}