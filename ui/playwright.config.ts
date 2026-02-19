/**
 * Playwright E2E Test Configuration
 *
 * CDR SOTA Requirements:
 * - E2E tests for claim→snippet→source navigation
 * - API integration verification
 * - UI responsiveness validation
 *
 * Run: npx playwright test
 * Debug: npx playwright test --ui
 */

import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',

  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  // Use Vite preview (production build) for E2E - supports SPA routing
  // In CI the app is already built; Playwright starts the preview server itself.
  webServer: {
    command: 'npx vite preview --host 0.0.0.0 --port 5173',
    url: 'http://localhost:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
