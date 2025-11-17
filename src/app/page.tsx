'use client';

import Link from "next/link";
import { ArrowRight, Brain, Zap, Target, Database, BarChart3, Sparkles, CheckCircle } from "lucide-react";
import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import ParticleBackground from "@/components/ParticleBackground";

export default function Home() {
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
  }, []);

  const features = [
    {
      icon: Database,
      title: "Smart Data Upload",
      description: "Upload your datasets with intelligent validation and preprocessing"
    },
    {
      icon: Brain,
      title: "AI Model Selection", 
      description: "Get personalized model recommendations based on your data characteristics"
    },
    {
      icon: Target,
      title: "Automated Training",
      description: "Train models with hyperparameter optimization and cross-validation"
    },
    {
      icon: BarChart3,
      title: "Real-time Analytics",
      description: "Monitor performance metrics and visualize results in real-time"
    }
  ];

  if (!isLoaded) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="spinner" />
      </div>
    );
  }

  return (
    <div className="min-h-screen relative overflow-hidden bg-gray-900">
      {/* Background Elements */}
      <ParticleBackground />
      <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-purple-900 to-cyan-900 opacity-20" />
      <div className="absolute inset-0 geometric-pattern opacity-30" />

      {/* Main Container */}
      <div className="relative z-10 min-h-screen">
        {/* Hero Section */}
        <section className="min-h-screen flex items-center justify-center" style={{paddingTop: '60px', paddingBottom: '60px'}}>
          <div className="container mx-auto max-w-6xl" style={{paddingLeft: '48px', paddingRight: '48px'}}>
            <div className="text-center" style={{display: 'flex', flexDirection: 'column', gap: '50px'}}>
              
              {/* Hero Badge */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                className="flex justify-center"
              >
                <div className="inline-flex w-auto items-center rounded-xl border border-cyan-500/30 bg-gray-800/50 backdrop-blur-xl" style={{paddingLeft: '20px', paddingRight: '20px'}}>
                  <Sparkles className="h-14 w-4 text-cyan-400" style={{marginRight: '8px'}} />
                  <span className="text-cyan-400 font-xl text-l">Next-Gen ML Platform</span>
                  {/* <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" style={{marginLeft: '8px'}} /> */}
                </div>
              </motion.div>

              {/* Brain Icon */}
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 1, delay: 0.4 }}
                className="flex justify-center"
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
                  <Brain className="h-16 w-16 sm:h-20 sm:w-20 lg:h-24 lg:w-24 text-cyan-400" />
                </motion.div>
              </motion.div>
              
              {/* Main Title */}
              <motion.div
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 1, delay: 0.6 }}
                style={{display: 'flex', flexDirection: 'column', gap: '24px'}}
              >
                <h1 className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-black leading-tight tracking-tight">
                  <span className="block text-white mb-2">Machine Learning</span>
                  <span 
                    className="block gradient-textS"
                    style={{
                      background: 'linear-gradient(135deg, #00f5ff 0%, #9945ff 100%)',
                      backgroundClip: 'text',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                    }}
                  >
                    Reimagined
                  </span>
                </h1>
                
                <p className="text-xl sm:text-2xl mt lg:text-2xl text-gray-300 leading-relaxed max-w-4xl mx-auto">
                  Harness the power of artificial intelligence with our cutting-edge platform. 
                  Upload your data, get instant AI recommendations, and deploy models in minutes.
                </p>
              </motion.div>

              {/* CTA Buttons */}
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 1.2 }}
                className="flex flex-col sm:flex-row gap-8 justify-center items-center"
              >
                <Link href="/upload">
                  <motion.button
                    className="flex items-center justify-center text-lg font-bold rounded-xl w-full sm:w-auto min-w-[220px]"
                    style={{
                      height: '70px',
                      paddingLeft: '40px', 
                      paddingRight: '40px', 
                      paddingTop: '20px', 
                      paddingBottom: '20px',
                      background: 'linear-gradient(135deg, #00f5ff 0%, #9945ff 100%)',
                      border: 'none',
                      color: '#ffffff',
                      fontWeight: '700',
                      textTransform: 'uppercase',
                      letterSpacing: '0.8px',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                      boxShadow: '0 8px 32px rgba(0, 245, 255, 0.3)',
                      position: 'relative',
                      overflow: 'hidden'
                    }}
                    whileHover={{ 
                      scale: 1.05,
                      boxShadow: '0 0 30px rgba(0, 245, 255, 0.4)'
                    }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Zap className="h-5 w-5" style={{marginRight: '8px'}} />
                    Start Building
                    <ArrowRight className="h-5 w-5" style={{marginLeft: '8px'}} />
                  </motion.button>
                </Link>
                
              
              </motion.div>

                {/* Decorative Horizontal Line with Sharp Edges */}
                <motion.div
                  initial={{ opacity: 0, scaleX: 0 }}
                  animate={{ opacity: 1, scaleX: 1 }}
                  transition={{ duration: 0.8, delay: 1.6 }}
                  className="relative flex items-center justify-center"
                  style={{ marginTop: '60px' }}
                >
                  <div className="relative w-[80vw] h-px">
                    {/* Main line */}
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-400 to-transparent" />
                    {/* Glowing effect */}
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-400/50 to-transparent blur-sm" />
                    {/* Sharp edges */}
                    <div className="absolute -left-1 top-1/2 transform -translate-y-1/2 w-2 h-2 bg-cyan-400 rotate-45" />
                    <div className="absolute -right-1 top-1/2 transform -translate-y-1/2 w-2 h-2 bg-cyan-400 rotate-45" />
                  </div>
                </motion.div>

              {/* Stats */}
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 1.4 }}
                className="" style={{paddingTop: '8px' }}
              >
                <div className="grid grid-cols-3 gap-16 max-w-3xl mx-auto">
                  {[
                    { value: "99.9%", label: "Accuracy" },
                    { value: "< 5min", label: "Setup Time" },
                    { value: "10K+", label: "Models Trained" }
                  ].map((stat, index) => (
                    <motion.div
                      key={stat.label}
                      className="text-center"
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.5, delay: 1.6 + index * 0.1 }}
                    >
                      <div className="text-2xl sm:text-3xl lg:text-4xl font-bold text-cyan-400" style={{marginBottom: '16px'}}>
                        {stat.value}
                      </div>
                      <div className="text-gray-400 text-sm sm:text-base">
                        {stat.label}
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            </div>
          </div>
        </section>

         {/* Decorative Horizontal Line with Sharp Edges */}
                <motion.div
                  initial={{ opacity: 0, scaleX: 0 }}
                  animate={{ opacity: 1, scaleX: 1 }}
                  transition={{ duration: 0.8, delay: 1.6 }}
                  className="relative flex items-center justify-center"
                  style={{    paddingBottom: '40px' }}
                >
                  <div className="relative w-[80vw] h-px">
                    {/* Main line */}
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-400 to-transparent" />
                    {/* Glowing effect */}
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-400/50 to-transparent blur-sm" />
                    {/* Sharp edges */}
                    <div className="absolute -left-1 top-1/2 transform -translate-y-1/2 w-2 h-2 bg-cyan-400 rotate-45" />
                    <div className="absolute -right-1 top-1/2 transform -translate-y-1/2 w-2 h-2 bg-cyan-400 rotate-45" />
                  </div>
                </motion.div>

        

        {/* Features Section */}
        <section style={{paddingTop:'10px', paddingLeft: '48px', paddingRight: '48px'}}>
          <div className="container mx-auto max-w-6xl">
            {/* Section Header */}
            <motion.div
              className="text-center space-y-8" style={{marginBottom: '100px'}}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white">
                Powerful Features
              </h2>
              <p className="text-md sm:text-2xl my-9 text-gray-300 leading-relaxed max-w-4xl mx-auto">
                Everything you need to build, train, and deploy machine learning models
              </p>
            </motion.div>

            {/* Features Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-12">
              {features.map((feature, index) => (
                <motion.div
                  key={feature.title}
                  className="group cursor-pointer"
                  initial={{ opacity: 0, y: 50 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  whileHover={{ y: -5 }}
                >
                  <div className="futuristic-card h-full text-center hover:border-cyan-400/50 transition-all duration-300" style={{padding: '40px'}}>
                    {/* Icon Container */}
                    <motion.div
                      className="flex items-center justify-center w-14 h-14 rounded-xl mx-auto"
                      style={{
                        background: 'linear-gradient(135deg, rgba(0, 245, 255, 0.2) 0%, rgba(153, 69, 255, 0.2) 100%)',
                        border: '1px solid rgba(0, 245, 255, 0.3)',
                        marginBottom: '32px'
                      }}
                      whileHover={{ 
                        scale: 1.1,
                        boxShadow: '0 0 20px rgba(0, 245, 255, 0.3)'
                      }}
                    >
                      <feature.icon className="h-7 w-7 text-cyan-400" />
                    </motion.div>
                    
                    {/* Content */}
                    <div className="space-y-6">
                      <h3 className="text-2xl font-semibold text-white">
                        {feature.title}
                      </h3>
                      <p className="text-gray-400 text-base leading-relaxed">
                        {feature.description}
                      </p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

       

        {/* Bottom Spacing */}
        <div style={{height: '160px'}}></div>
      </div>
    </div>
  );
}
