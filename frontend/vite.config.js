import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  server: {
    port: 3000,
    open: true,
  },
  build: {
    outDir: "build",
  },
  esbuild: {
    loader: "jsx",
    include: /src\/.*\.jsx?$/,
  },
  optimizeDeps: {
    esbuild: {
      loader: { ".js": "jsx" },
    },
  },
});
