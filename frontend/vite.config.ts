/// <reference types="vitest/config" />
import type { Plugin } from 'vite'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const apiProxyTarget = process.env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000'
const threeCdnUrl = 'https://cdn.jsdelivr.net/npm/three@0.185.1/build/three.module.js'

function threeCdnPlugin(): Plugin {
  return {
    name: 'three-cdn',
    enforce: 'pre',
    resolveId(source) {
      if (source === 'three') {
        return { id: threeCdnUrl, external: true }
      }
    },
  }
}

export default defineConfig(({ mode }) => ({
  plugins: [mode !== 'test' && threeCdnPlugin(), vue()],
  resolve: {
    extensions: ['.ts', '.tsx', '.mjs', '.js', '.jsx', '.json'],
    alias: {
      '@': '/src',
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: apiProxyTarget,
        changeOrigin: true,
      },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    include: ['src/**/*.spec.ts'],
  },
}))
