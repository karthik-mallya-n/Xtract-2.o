'use client';

import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Points, PointMaterial } from '@react-three/drei';
import * as THREE from 'three';

interface ParticleFieldProps {
  count?: number;
}

function ParticleField({ count = 3000 }: ParticleFieldProps) {
  const ref = useRef<THREE.Points>(null!);
  
  // Generate random positions for particles with better distribution
  const positions = useMemo(() => {
    const positions = new Float32Array(count * 3);
    
    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      
      // Spread particles in a large area
      positions[i3] = (Math.random() - 0.5) * 40;
      positions[i3 + 1] = (Math.random() - 0.5) * 40;
      positions[i3 + 2] = (Math.random() - 0.5) * 40;
    }
    
    return positions;
  }, [count]);

  // Optimize animation with smoother, less intensive calculations
  useFrame((state) => {
    if (ref.current) {
      const time = state.clock.getElapsedTime();
      
      // Smooth, slow rotation for mesmerizing effect
      ref.current.rotation.x = time * 0.05;
      ref.current.rotation.y = time * 0.03;
      
      // Gentle floating motion
      ref.current.position.y = Math.sin(time * 0.4) * 0.1;
    }
  });

  return (
    <group rotation={[0, 0, Math.PI / 4]}>
      <Points ref={ref} positions={positions} stride={3} frustumCulled>
        <PointMaterial
          transparent
          color="#00f5ff"
          size={0.02}
          sizeAttenuation={true}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          opacity={0.8}
        />
      </Points>
    </group>
  );
}

export default function ParticleBackground() {
  // Detect if user prefers reduced motion
  const prefersReducedMotion = typeof window !== 'undefined' && 
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (prefersReducedMotion) {
    // Return a static background for users who prefer reduced motion
    return (
      <div className="absolute inset-0 -z-10">
        <div 
          className="w-full h-full opacity-20"
          style={{
            background: 'radial-gradient(circle at 50% 50%, rgba(0, 245, 255, 0.1) 0%, transparent 70%)'
          }}
        />
      </div>
    );
  }

  return (
    <div className="absolute inset-0 -z-10 opacity-70">
      <Canvas
        camera={{ position: [0, 0, 1], fov: 75 }}
        style={{ background: 'transparent' }}
        gl={{ 
          antialias: false, // Disabled for better performance
          powerPreference: "high-performance",
          alpha: true,
          preserveDrawingBuffer: false
        }}
        dpr={Math.min(typeof window !== 'undefined' ? window.devicePixelRatio : 1, 2)}
        performance={{ min: 0.5 }}
      >
        <ParticleField count={3000} />
      </Canvas>
    </div>
  );
}