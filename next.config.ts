import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  reactStrictMode: true,
  // Suppress warnings about version mismatches
  onDemandEntries: {
    maxInactiveAge: 25 * 1000,
    pagesBufferLength: 2,
  },
  // Webpack config to handle version parsing issues
  webpack: (config, { isServer }) => {
    // Suppress semver parsing warnings and errors
    config.ignoreWarnings = [
      ...(config.ignoreWarnings || []),
      { module: /node_modules/ },
      { message: /Invalid argument not valid semver/ },
      { message: /not valid semver/ },
    ];
    
    // Handle resolve fallbacks
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
      };
    }
    return config;
  },
  // Logging configuration
  logging: {
    fetches: {
      fullUrl: false,
    },
  },
};

export default nextConfig;
