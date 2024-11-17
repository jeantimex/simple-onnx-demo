import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/onnxruntime-web/dist/*.wasm',
          dest: 'public'
        },
        {
          src: 'node_modules/onnxruntime-web/dist/webgpu/*.wasm',
          dest: 'public/webgpu'
        }
      ]
    })
  ],
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },
  server: {
    port: 3000,
    open: true,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    }
  },
  build: {
    target: 'esnext',
    rollupOptions: {
      output: {
        manualChunks: {
          onnx: ['onnxruntime-web']
        }
      }
    }
  }
});