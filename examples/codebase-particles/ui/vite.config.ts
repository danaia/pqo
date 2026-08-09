import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  base: './',
  plugins: [vue()],
  build: {
    target: 'safari14',
    cssCodeSplit: false,
    sourcemap: false,
  },
})
