/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the page is served by Vite on :5173 and FastAPI runs on :8000.
// The proxy forwards API calls so the browser only ever talks to one origin,
// which means no CORS setup is needed (same as production, where FastAPI
// serves both the page and the API).
const API_TARGET = "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/chat": API_TARGET,
      "/health": API_TARGET,
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/setupTests.ts"],
  },
});
