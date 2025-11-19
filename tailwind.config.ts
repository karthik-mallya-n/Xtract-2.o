import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  mode: 'jit',
  theme: {
    extend: {
      colors: {
        // Add custom colors if needed
        'neon-cyan': '#00f5ff',
        'neon-purple': '#9945ff',
        'neon-pink': '#ff0080',
        'neon-green': '#00ff88',
        'neon-blue': '#0066ff',
      },
      fontFamily: {
        'inter': ['Inter', 'sans-serif'],
        'mono': ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'spin': 'spin 1s linear infinite',
        'pulse': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce': 'bounce 1s infinite',
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem',
      },
      fontSize: {
        'xxs': '0.625rem',
        '8xl': '6rem',
        '9xl': '8rem',
      },
    },
  },
  plugins: [],
}

export default config