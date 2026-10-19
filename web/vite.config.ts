import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:7860',
      '/health': 'http://127.0.0.1:7860'
    }
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/testSetup.ts'
  }
})
