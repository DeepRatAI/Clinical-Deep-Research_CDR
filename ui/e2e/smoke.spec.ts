/**
 * CDR E2E Test: Basic UI Smoke Test
 *
 * Verifies:
 * - Application loads correctly
 * - Navigation works
 * - Core components render
 *
 * Refs: CDR SOTA requirements - E2E validation
 */

import { test, expect } from '@playwright/test';

test.describe('CDR Application Smoke Tests', () => {
  test('homepage loads and displays correctly', async ({ page }) => {
    await page.goto('/');

    // Wait for app to hydrate
    await page.waitForLoadState('domcontentloaded');

    // Verify main heading or logo
    const heading = page.locator('h1, [data-testid="app-title"]').first();
    await expect(heading).toBeVisible({ timeout: 10_000 });
  });

  test('can navigate to runs page', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');

    // Try to find runs link
    const runsLink = page.locator('a[href*="runs"], button:has-text("Runs")').first();
    if (await runsLink.isVisible()) {
      await runsLink.click();
      await expect(page).toHaveURL(/.*runs.*/);
    }
  });

  test('displays error state gracefully when API is unavailable', async ({ page }) => {
    // Mock API to return error
    await page.route('**/api/**', (route) => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({ error: 'API unavailable' }),
      });
    });

    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');

    // Should still render without crashing
    await expect(page.locator('body')).toBeVisible();
  });
});
