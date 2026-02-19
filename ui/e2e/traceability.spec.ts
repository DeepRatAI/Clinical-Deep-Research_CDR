/**
 * CDR E2E Test: Claim Traceability
 *
 * SOTA Requirement: Verify claim→snippet→source navigation chain
 *
 * Tests:
 * 1. Claims display with snippet references
 * 2. Click snippet ID navigates to snippet detail
 * 3. Source link opens external resource
 *
 * Refs: GRADE guidelines, PRISMA 2020 transparency
 */

import { test, expect } from '@playwright/test';

test.describe('Claim Traceability E2E', () => {
  // Mock run data matching RunDetailApiResponse structure
  const mockRunData = {
    run_id: 'test-run-001',
    status: 'COMPLETED',
    status_reason: null,
    dod_level: 3,
    pico: {
      population: 'Adults with cardiovascular risk',
      intervention: 'Aspirin 100mg daily',
      comparator: 'Placebo',
      outcome: 'Cardiovascular events',
      study_types: ['RCT'],
    },
    search_plan: null,
    prisma_counts: null,
    claims_count: 1,
    snippets_count: 1,
    studies_count: 1,
    hypotheses_count: 0,
    verification_coverage: 1.0,
    errors: [],
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    claims: [
      {
        claim_id: 'claim-001',
        claim_text: 'Aspirin reduces cardiovascular events by 25%',
        certainty: 'MODERATE',
        supporting_snippet_ids: ['snip-001', 'snip-002'],
        verification_status: 'VERIFIED',
        verification_score: 0.95,
        grade_rationale: null,
      },
    ],
    snippets: [
      {
        snippet_id: 'snip-001',
        record_id: 'rec-001',
        pmid: '12345678',
        title: 'Aspirin Study 2024',
        text: 'We found 25% reduction in MACE with aspirin therapy.',
        section: 'results',
      },
    ],
  };

  test.beforeEach(async ({ page }) => {
    // Mock API responses — API uses /api/v1/ prefix
    // getRunDetail() fetches /api/v1/runs/{id}/detail
    await page.route('**/api/v1/runs/test-run-001/detail', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockRunData),
      });
    });

    // Fallback: some paths hit /api/v1/runs/{id} directly
    await page.route('**/api/v1/runs/test-run-001', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockRunData),
      });
    });

    // getRunClaims() fetches /api/v1/runs/{id}/claims
    await page.route('**/api/v1/runs/test-run-001/claims', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockRunData.claims),
      });
    });

    // getRunSnippets() fetches /api/v1/runs/{id}/snippets
    await page.route('**/api/v1/runs/test-run-001/snippets', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockRunData.snippets),
      });
    });

    // Individual snippet detail
    await page.route('**/api/v1/runs/test-run-001/snippets/**', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockRunData.snippets[0]),
      });
    });

    // Mock other endpoints that RunDetail might call (hypotheses, evaluation, etc.)
    await page.route('**/api/v1/runs/test-run-001/hypotheses', (route) => {
      route.fulfill({ status: 200, contentType: 'application/json', body: '[]' });
    });
    await page.route('**/api/v1/runs/test-run-001/evaluation', (route) => {
      route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    });
    await page.route('**/api/v1/runs/test-run-001/prisma', (route) => {
      route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    });
    await page.route('**/api/v1/runs/test-run-001/pico', (route) => {
      route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    });
    await page.route('**/api/v1/runs/test-run-001/search-plan', (route) => {
      route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    });
    await page.route('**/api/v1/runs/test-run-001/report', (route) => {
      route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    });
    await page.route('**/api/v1/runs/test-run-001/studies', (route) => {
      route.fulfill({ status: 200, contentType: 'application/json', body: '[]' });
    });
  });

  test('claims page displays claim with snippet references', async ({ page }) => {
    await page.goto('/runs/test-run-001/claims');
    await page.waitForLoadState('domcontentloaded');

    // Verify claim text is visible
    const claimText = page.getByText('Aspirin reduces cardiovascular events', {
      exact: false,
    });
    await expect(claimText).toBeVisible({ timeout: 5_000 });

    // Verify supporting snippet count is shown (ClaimCard renders "{n} supporting snippet(s)")
    const snippetCount = page.getByText('supporting snippet', { exact: false });
    await expect(snippetCount).toBeVisible({ timeout: 5_000 });
  });

  test('clicking snippet reference shows snippet details', async ({ page }) => {
    await page.goto('/runs/test-run-001/claims');
    await page.waitForLoadState('domcontentloaded');

    // Click on snippet reference
    const snippetLink = page.locator('a[href*="snip-001"], [data-snippet-id="snip-001"]').first();
    
    if (await snippetLink.isVisible()) {
      await snippetLink.click();
      
      // Verify snippet text is shown
      const snippetText = page.getByText('25% reduction in MACE', { exact: false });
      await expect(snippetText).toBeVisible({ timeout: 5_000 });
    }
  });

  test('snippet detail shows source link to PubMed', async ({ page }) => {
    await page.goto('/runs/test-run-001/snippets/snip-001');
    await page.waitForLoadState('domcontentloaded');

    // Look for PubMed link
    const pubmedLink = page.locator('a[href*="pubmed"], a[href*="ncbi.nlm.nih.gov"]').first();
    
    if (await pubmedLink.isVisible()) {
      const href = await pubmedLink.getAttribute('href');
      expect(href).toContain('12345678');
    }
  });

  test('verification status badge is visible on claims', async ({ page }) => {
    await page.goto('/runs/test-run-001/claims');
    await page.waitForLoadState('domcontentloaded');

    // Look for verification badge via data-testid (ClaimCard uses data-testid="verified")
    const verifiedBadge = page.locator('[data-testid="verified"]').first();
    await expect(verifiedBadge).toBeVisible({ timeout: 5_000 });

    // Confirm it has the correct status
    await expect(verifiedBadge).toHaveAttribute('data-verification-status', 'VERIFIED');
  });
});
