import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
// The site reads the committed results_*.json, the artifact manifest and src/generated/*.json
// at build time, so every number on the page is generated from repo files, never typed.
export default defineConfig({
  base: '/signal-ml-lab/',
  plugins: [react()],
  server: { fs: { allow: ['..'] } },
});
