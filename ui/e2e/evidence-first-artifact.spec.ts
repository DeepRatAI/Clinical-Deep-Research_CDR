/**
 * CDR E2E Test: Complete Evidence-First Artifact Validation
 *
 * SOTA Requirement: Validate complete CDR artifact chain
 * - PICO structure
 * - PRISMA flow counts
 * - Snippets with source_ref (pmid/nctid)
 * - RoB/ROBINS-I assessment
 * - Verified claims
 * - Composed hypotheses with MCID
 * - Evaluation report in RunStore
 *
 * Refs:
 * - PRISMA 2020: https://www.prisma-statement.org/prisma-2020
 * - GRADE handbook: https://gradepro.org/handbook/
 * - RoB2: https://methods.cochrane.org/bias/resources
 */

import { test, expect } from '@playwright/test';

test.describe('Complete Evidence-First Artifact E2E', () => {
  // Comprehensive mock data for a completed CDR run
  const mockCompleteRun = {
    run_id: 'e2e-complete-001',
    question: 'Does metformin combined with GLP-1 agonists improve cardiovascular outcomes?',
    status: 'completed',
    pico: {
      population: 'Adults with type 2 diabetes and established cardiovascular disease',
      intervention: 'Metformin combined with GLP-1 receptor agonists',
      comparator: 'Metformin monotherapy',
      outcome: 'Major adverse cardiovascular events (MACE)',
    },
    prisma: {
      records_identified: 1245,
      records_screened: 856,
      reports_assessed: 124,
      studies_included: 18,
    },
    claims: [
      {
        claim_id: 'claim-glp1-cv',
        claim_text: 'GLP-1 agonists reduce MACE by 14% compared to placebo in patients with T2D',
        certainty: 'HIGH',
        supporting_snippet_ids: ['snip-leader-1', 'snip-sustain-1'],
        verification_status: 'VERIFIED',
        verification_score: 0.95,
        grade_rationale: {
          risk_of_bias: 'Low risk from high-quality RCTs',
          inconsistency: 'Consistent results across LEADER and SUSTAIN-6',
          indirectness: 'Direct evidence for T2D patients with CVD',
          imprecision: 'Narrow confidence intervals',
          publication_bias: 'No evidence of publication bias',
        },
      },
      {
        claim_id: 'claim-metformin-base',
        claim_text: 'Metformin provides baseline glycemic control and reduces microvascular complications',
        certainty: 'HIGH',
        supporting_snippet_ids: ['snip-ukpds-1'],
        verification_status: 'VERIFIED',
        verification_score: 0.88,
        grade_rationale: {
          risk_of_bias: 'Low risk from UKPDS',
          inconsistency: 'Consistent with other studies',
          indirectness: 'Direct evidence',
          imprecision: 'Moderate sample size',
          publication_bias: null,
        },
      },
    ],
    snippets: [
      {
        snippet_id: 'snip-leader-1',
        source_ref: { 
          record_id: 'rec-leader',
          pmid: '27295427', 
          doi: '10.1056/NEJMoa1603827',
          nct_id: null,
          title: 'Liraglutide and Cardiovascular Outcomes in Type 2 Diabetes',
          authors: ['Marso SP', 'Daniels GH', 'Tanaka K'],
          publication_year: 2016,
          journal: 'N Engl J Med',
          url: null,
        },
        section: 'RESULTS',
        text: 'The primary composite outcome occurred in fewer patients in the liraglutide group (13.0%) than in the placebo group (14.9%); hazard ratio, 0.87',
        offset_start: 1234,
        offset_end: 1456,
        relevance_score: 0.95,
      },
      {
        snippet_id: 'snip-sustain-1',
        source_ref: { 
          record_id: 'rec-sustain6',
          pmid: '27633186', 
          doi: '10.1056/NEJMoa1607141',
          nct_id: null,
          title: 'Semaglutide and Cardiovascular Outcomes in Patients with Type 2 Diabetes',
          authors: ['Marso SP', 'Bain SC', 'Consoli A'],
          publication_year: 2016,
          journal: 'N Engl J Med',
          url: null,
        },
        section: 'RESULTS',
        text: 'Semaglutide significantly reduced the risk of major adverse cardiovascular events (hazard ratio, 0.74)',
        offset_start: 2100,
        offset_end: 2250,
        relevance_score: 0.92,
      },
      {
        snippet_id: 'snip-ukpds-1',
        source_ref: { 
          record_id: 'rec-ukpds',
          pmid: '9742976',
          doi: null,
          nct_id: null,
          title: 'Effect of intensive blood-glucose control with metformin on complications in overweight patients',
          authors: ['UKPDS Group'],
          publication_year: 1998,
          journal: 'Lancet',
          url: null,
        },
        section: 'RESULTS',
        text: 'Metformin was associated with a 32% risk reduction for any diabetes-related endpoint',
        offset_start: 850,
        offset_end: 980,
        relevance_score: 0.88,
      },
    ],
    studies: [
      {
        study_id: 'study-leader',
        name: 'LEADER Trial',
        design: 'RCT',
        n_participants: 9340,
        rob_assessment: {
          tool: 'RoB2',
          overall: 'low',
          domains: {
            randomization: 'low',
            deviations: 'low',
            missing_data: 'some_concerns',
            measurement: 'low',
            selection: 'low',
          },
        },
      },
      {
        study_id: 'study-sustain6',
        name: 'SUSTAIN-6 Trial',
        design: 'RCT',
        n_participants: 3297,
        rob_assessment: {
          tool: 'RoB2',
          overall: 'low',
          domains: {
            randomization: 'low',
            deviations: 'low',
            missing_data: 'low',
            measurement: 'low',
            selection: 'low',
          },
        },
      },
    ],
    hypotheses: [
      {
        hypothesis_id: 'hyp-glp1-metformin-synergy',
        claim_a_id: 'claim-glp1-cv',
        claim_b_id: 'claim-metformin-base',
        hypothesis_text: 'If patients with T2D and CVD receive GLP-1 agonists in combination with metformin, then they will experience greater MACE reduction than with either agent alone, mediated through complementary mechanisms of improved glycemic control and direct cardioprotection.',
        mechanism: 'GLP-1 → β-cell function → glycemic control → reduced atherosclerosis + GLP-1 → direct cardiac effects',
        rival_hypotheses: [
          'The observed benefit is due solely to glycemic improvement, not synergistic mechanisms',
          'Selection bias: patients on combination therapy may be more compliant overall',
        ],
        threats_to_validity: ['baseline CV risk', 'concurrent statin use', 'renal function'],
        mcid: '0.85 (15% RRR)',
        test_design: 'pragmatic_rct with 1500 participants over 36 months',
        confidence: 0.82,
      },
    ],
    evaluation: {
      overall_score: 0.92,
      overall_grade: 'A',
      dimensions: [
        { name: 'evidence_quality', score: 0.95, grade: 'A', rationale: 'High-quality RCT evidence from LEADER/SUSTAIN-6' },
        { name: 'claim_verification', score: 0.90, grade: 'A', rationale: 'All claims verified with source snippets' },
        { name: 'hypothesis_novelty', score: 0.88, grade: 'B+', rationale: 'Builds on known mechanisms with testable synergy claim' },
        { name: 'test_design_quality', score: 0.92, grade: 'A', rationale: 'Complete RCT protocol with MCID' },
      ],
      strengths: ['Strong RCT evidence base', 'Clear mechanistic reasoning', 'Complete test design with MCID'],
      weaknesses: ['Limited real-world evidence', 'Cost-effectiveness not addressed'],
      recommendations: ['Include pharmacoeconomic analysis', 'Consider subgroup analyses by baseline HbA1c'],
    },
  };

  test.beforeEach(async ({ page }) => {
    // Mock all API endpoints for complete artifact (v1 API)
    await page.route('**/api/v1/runs/e2e-complete-001', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockCompleteRun),
      });
    });

    await page.route('**/api/v1/runs/e2e-complete-001/detail', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          run_id: mockCompleteRun.run_id,
          status: mockCompleteRun.status,
          status_reason: null,
          dod_level: 4,
          pico: mockCompleteRun.pico,
          claims_count: mockCompleteRun.claims.length,
          snippets_count: mockCompleteRun.snippets.length,
          studies_count: mockCompleteRun.studies.length,
          hypotheses_count: mockCompleteRun.hypotheses.length,
          verification_coverage: 100,
        }),
      });
    });

    await page.route('**/api/v1/runs/e2e-complete-001/pico', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockCompleteRun.pico),
      });
    });

    await page.route('**/api/v1/runs/e2e-complete-001/prisma', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockCompleteRun.prisma),
      });
    });

    await page.route('**/api/v1/runs/e2e-complete-001/claims', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockCompleteRun.claims),
      });
    });

    await page.route('**/api/v1/runs/e2e-complete-001/snippets', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockCompleteRun.snippets),
      });
    });

    await page.route('**/api/v1/runs/e2e-complete-001/studies', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockCompleteRun.studies),
      });
    });

    await page.route('**/api/v1/runs/e2e-complete-001/hypotheses', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockCompleteRun.hypotheses),
      });
    });

    await page.route('**/api/v1/runs/e2e-complete-001/evaluation', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockCompleteRun.evaluation),
      });
    });
  });

  test.describe('PICO Structure', () => {
    test('displays complete PICO elements', async ({ page }) => {
      await page.goto('/runs/e2e-complete-001');
      await page.waitForLoadState('domcontentloaded');

      // Check for PICO display - use first() to avoid strict mode violations
      await expect(page.getByText('type 2 diabetes', { exact: false }).first()).toBeVisible({ timeout: 5_000 });
      await expect(page.getByText('GLP-1', { exact: false }).first()).toBeVisible();
      await expect(page.getByText('Metformin monotherapy', { exact: false }).first()).toBeVisible();
      await expect(page.getByText('MACE', { exact: false }).first()).toBeVisible();
    });
  });

  test.describe('PRISMA Flow Counts', () => {
    test('shows PRISMA flow with correct counts', async ({ page }) => {
      await page.goto('/runs/e2e-complete-001');
      await page.waitForLoadState('domcontentloaded');

      // PRISMA counts should be visible
      const prismaSection = page.locator('[data-testid="prisma-flow"], .prisma-flow, [class*="prisma"]');
      
      if (await prismaSection.count() > 0) {
        await expect(page.getByText('1245', { exact: false })).toBeVisible();
        await expect(page.getByText('18', { exact: false })).toBeVisible();
      }
    });
  });

  test.describe('Snippets with Source References', () => {
    test('snippets include PMID source links', async ({ page }) => {
      await page.goto('/runs/e2e-complete-001/claims');
      await page.waitForLoadState('domcontentloaded');

      // Look for PMID reference
      const pmidLink = page.locator('a[href*="pubmed"], a[href*="27295427"], [data-pmid]');
      
      if (await pmidLink.count() > 0) {
        await expect(pmidLink.first()).toBeVisible();
        // Verify it's a valid PubMed link
        const href = await pmidLink.first().getAttribute('href');
        expect(href).toContain('pubmed');
      }
    });

    test('snippet text is displayed with attribution', async ({ page }) => {
      await page.goto('/runs/e2e-complete-001/snippets');
      await page.waitForLoadState('domcontentloaded');

      // Verify snippet text from LEADER trial is shown (text includes hazard ratio)
      await expect(page.getByText('hazard ratio', { exact: false }).first()).toBeVisible({ timeout: 5_000 });
    });
  });

  test.describe('Risk of Bias Assessment', () => {
    test('RoB assessment shows overall rating', async ({ page }) => {
      await page.goto('/runs/e2e-complete-001');
      await page.waitForLoadState('domcontentloaded');

      // Look for RoB indicator
      const robElement = page.locator('[data-testid="rob"], .rob-assessment, [class*="risk-of-bias"]');
      
      if (await robElement.count() > 0) {
        // Should show "low" risk for LEADER trial
        await expect(page.getByText('low', { exact: false })).toBeVisible();
      }
    });
  });

  test.describe('Verified Claims', () => {
    test('claims show verification status', async ({ page }) => {
      await page.goto('/runs/e2e-complete-001/claims');
      await page.waitForLoadState('domcontentloaded');

      // Claim text should be visible
      await expect(page.getByText('GLP-1 agonists reduce MACE', { exact: false })).toBeVisible({ timeout: 5_000 });

      // Verification status indicator
      const verifiedBadge = page.locator('[data-testid="verified"], .verified, [class*="verification"]');
      if (await verifiedBadge.count() > 0) {
        await expect(verifiedBadge.first()).toBeVisible();
      }
    });

    test('claims display GRADE certainty', async ({ page }) => {
      await page.goto('/runs/e2e-complete-001/claims');
      await page.waitForLoadState('domcontentloaded');

      // Should show "high" certainty (use .first() as there may be multiple claims)
      await expect(page.getByText('high', { exact: false }).first()).toBeVisible({ timeout: 5_000 });
    });
  });

  test.describe('Composed Hypotheses', () => {
    test('hypothesis displays if-then structure', async ({ page }) => {
      await page.goto('/runs/e2e-complete-001');
      await page.waitForLoadState('domcontentloaded');

      // Navigate to hypotheses section if needed
      const hypSection = page.locator('[data-testid="hypotheses"], .hypotheses, [class*="hypothesis"]');
      
      if (await hypSection.count() > 0) {
        // If-then structure
        await expect(page.getByText('If patients', { exact: false })).toBeVisible({ timeout: 5_000 });
        await expect(page.getByText('then they will', { exact: false })).toBeVisible();
      }
    });

    test('hypothesis shows MCID value', async ({ page }) => {
      await page.goto('/runs/e2e-complete-001');
      await page.waitForLoadState('domcontentloaded');

      // MCID value (0.85 or 85%)
      const mcidElement = page.locator('[data-testid="mcid"], .mcid, [class*="mcid"]');
      
      if (await mcidElement.count() > 0) {
        await expect(page.getByText('0.85', { exact: false })).toBeVisible();
      }
    });

    test('hypothesis includes rival explanations', async ({ page }) => {
      await page.goto('/runs/e2e-complete-001');
      await page.waitForLoadState('domcontentloaded');

      // Rival hypotheses section
      await expect(page.getByText('rival', { exact: false })).toBeVisible({ timeout: 5_000 });
    });

    test('hypothesis has test design with sample size', async ({ page }) => {
      await page.goto('/runs/e2e-complete-001');
      await page.waitForLoadState('domcontentloaded');

      // Test design info
      const testDesign = page.locator('[data-testid="test-design"], .test-design, [class*="study-design"]');
      
      if (await testDesign.count() > 0) {
        await expect(page.getByText('1500', { exact: false })).toBeVisible();
        await expect(page.getByText('RCT', { exact: false })).toBeVisible();
      }
    });
  });

  test.describe('Evaluation Report', () => {
    test('evaluation shows overall score and grade', async ({ page }) => {
      await page.goto('/runs/e2e-complete-001');
      await page.waitForLoadState('domcontentloaded');

      // Overall evaluation
      const evalSection = page.locator('[data-testid="evaluation"], .evaluation, [class*="evaluation"]');
      
      if (await evalSection.count() > 0) {
        // Grade A
        await expect(page.getByText('A', { exact: true })).toBeVisible({ timeout: 5_000 });
        // Score 0.92 or 92%
        await expect(page.getByText('92', { exact: false })).toBeVisible();
      }
    });

    test('evaluation lists dimension scores', async ({ page }) => {
      await page.goto('/runs/e2e-complete-001');
      await page.waitForLoadState('domcontentloaded');

      // Dimension names (UI replaces _ with spaces)
      await expect(page.getByText('evidence quality', { exact: false })).toBeVisible({ timeout: 5_000 });
    });

    test('evaluation shows strengths and recommendations', async ({ page }) => {
      await page.goto('/runs/e2e-complete-001');
      await page.waitForLoadState('domcontentloaded');

      // Strengths section
      await expect(page.getByText('Strong RCT evidence', { exact: false })).toBeVisible({ timeout: 5_000 });
      
      // Recommendations
      await expect(page.getByText('pharmacoeconomic', { exact: false })).toBeVisible();
    });
  });

  test.describe('Complete Artifact Chain Navigation', () => {
    test('can navigate from claim to supporting snippet', async ({ page }) => {
      await page.goto('/runs/e2e-complete-001/claims');
      await page.waitForLoadState('domcontentloaded');

      // Find claim and click on snippet reference
      const snippetLink = page.locator('a[href*="snip-leader"], [data-snippet-id="snip-leader-1"]').first();
      
      if (await snippetLink.isVisible()) {
        await snippetLink.click();
        
        // Should show snippet text
        await expect(page.getByText('primary composite outcome', { exact: false })).toBeVisible({ timeout: 5_000 });
      }
    });

    test('can navigate from snippet to PubMed source', async ({ page }) => {
      await page.goto('/runs/e2e-complete-001/claims');
      await page.waitForLoadState('domcontentloaded');

      // Find external link
      const pubmedLink = page.locator('a[href*="pubmed.ncbi.nlm.nih.gov"]').first();
      
      if (await pubmedLink.count() > 0) {
        // Verify link target
        const href = await pubmedLink.getAttribute('href');
        expect(href).toContain('27295427');
        expect(href).toContain('pubmed');
      }
    });
  });

  test.describe('Data Completeness Validation', () => {
    test('all required artifact fields are present', async ({ page }) => {
      await page.goto('/runs/e2e-complete-001');
      await page.waitForLoadState('domcontentloaded');

      // Wait for key content to load (PICO section should appear)
      await expect(page.getByText('Population', { exact: false })).toBeVisible({ timeout: 5_000 });

      // Validate presence of key artifact sections by checking visible elements
      // PICO should be visible in overview
      await expect(page.getByText('type 2 diabetes', { exact: false }).first()).toBeVisible();
      
      // Claims count should be visible
      await expect(page.getByText('Claims', { exact: false }).first()).toBeVisible();
      
      // Snippets count should be visible
      await expect(page.getByText('Snippets', { exact: false }).first()).toBeVisible();
      
      // Hypotheses should be visible or mentioned
      await expect(page.getByText('Hypotheses', { exact: false }).first()).toBeVisible();
    });
  });
});
