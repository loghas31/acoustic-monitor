import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Static build (Vercel/Netlify) hitting the FastAPI backend; the API base URL
// comes from VITE_API_URL at build time. Dev proxy avoids CORS noise locally.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": { target: "http://localhost:8000", rewrite: p => p.replace(/^\/api/, "") } },
  },
});
