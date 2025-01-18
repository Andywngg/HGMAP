module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          900: '#2D0000',
          800: '#400000',
          700: '#660000',
          600: '#8B0000', // Dark Red
          500: '#B00000',
          400: '#D40000',
        },
        dark: {
          900: '#000000',
          800: '#1A1A1A',
          700: '#2D2D2D',
          600: '#404040',
        }
      },
    },
  },
  plugins: [],
} 