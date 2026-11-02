import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  base: "/v1/console/",
  plugins: [react()],
  build: {
    outDir: resolve(__dirname, "../static/console"),
    emptyOutDir: true,
    assetsDir: "assets",
  },
});
