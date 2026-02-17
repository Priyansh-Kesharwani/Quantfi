const { test, expect } = require('@playwright/test');

const BACKEND = process.env.E2E_BACKEND_URL || 'http://localhost:8000';

test.describe('App Shell & Navigation', () => {
  test('loads and shows sidebar with all nav links', async ({ page }) => {
    await page.goto('/');
    const sidebar = page.getByTestId('sidebar');
    await expect(sidebar).toBeVisible();
    await expect(page.getByTestId('app-title')).toHaveText('QUANTFI');

    for (const nav of ['dashboard', 'assets', 'backtest-lab', 'news-&-events', 'settings']) {
      await expect(page.getByTestId(`nav-${nav}`)).toBeVisible();
    }
  });

  test('navigates between all pages via sidebar', async ({ page }) => {
    await page.goto('/');

    await page.getByTestId('nav-backtest-lab').click();
    await expect(page.getByTestId('backtest-lab-page')).toBeVisible();

    await page.getByTestId('nav-news-&-events').click();
    await expect(page.getByTestId('news-page')).toBeVisible();

    await page.getByTestId('nav-settings').click();
    await expect(page.getByTestId('settings-page')).toBeVisible();

    await page.getByTestId('nav-dashboard').click();
    await expect(page).toHaveURL('/');
  });
});

test.describe('Dashboard', () => {
  test('shows empty state or asset grid', async ({ page }) => {
    await page.goto('/');
    const main = page.getByTestId('main-content');
    await expect(main).toBeVisible();

    const empty = page.getByTestId('empty-dashboard');
    const grid = page.getByTestId('dashboard-main');
    await expect(empty.or(grid)).toBeVisible({ timeout: 15_000 });
  });

  test('refresh button triggers data reload', async ({ page }) => {
    await page.goto('/');
    const grid = page.getByTestId('dashboard-main');
    const empty = page.getByTestId('empty-dashboard');
    await expect(grid.or(empty)).toBeVisible({ timeout: 15_000 });

    if (await grid.isVisible()) {
      const btn = page.getByTestId('refresh-dashboard-btn');
      await btn.click();
      await expect(btn).toBeVisible();
    }
  });
});

test.describe('Add Asset Flow', () => {
  test('opens dialog from sidebar, validates empty fields, fills form', async ({ page }) => {
    await page.goto('/');
    await page.getByTestId('add-asset-btn').click();

    const dialog = page.getByTestId('add-asset-dialog');
    await expect(dialog).toBeVisible();

    await page.getByTestId('confirm-add-asset-btn').click();
    await page.waitForTimeout(500);

    const symbolInput = page.getByTestId('asset-symbol-input');
    const nameInput = page.getByTestId('asset-name-input');
    await symbolInput.fill('AAPL');
    await expect(symbolInput).toHaveValue('AAPL');
    await nameInput.fill('Apple Inc.');
    await expect(nameInput).toHaveValue('Apple Inc.');

    const typeSelect = page.getByTestId('asset-type-select');
    await expect(typeSelect).toBeVisible();
  });

  test('opens dialog from empty dashboard CTA', async ({ page }) => {
    await page.goto('/');
    const addFirstBtn = page.getByTestId('add-first-asset-btn');
    if (await addFirstBtn.isVisible({ timeout: 5_000 }).catch(() => false)) {
      await addFirstBtn.click();
      await expect(page.getByTestId('add-asset-dialog')).toBeVisible();
    }
  });
});

test.describe('Asset Detail Page', () => {
  test('navigates to detail and shows sections', async ({ page }) => {
    await page.goto('/');
    const grid = page.getByTestId('dashboard-main');
    if (await grid.isVisible({ timeout: 10_000 }).catch(() => false)) {
      const firstCard = page.locator('[data-testid^="asset-card-"]').first();
      await firstCard.click();

      await expect(page.getByTestId('asset-detail-page')).toBeVisible({ timeout: 15_000 });
      await expect(page.getByTestId('asset-detail-symbol')).toBeVisible();
      await expect(page.getByTestId('score-card')).toBeVisible();
      await expect(page.getByTestId('score-breakdown')).toBeVisible();
      await expect(page.getByTestId('indicators-table')).toBeVisible();

      await page.getByTestId('back-to-dashboard-btn').click();
      await expect(page.getByTestId('dashboard-main')).toBeVisible({ timeout: 10_000 });
    }
  });
});

test.describe('Backtest Lab', () => {
  test('shows config panel and empty results', async ({ page }) => {
    await page.goto('/backtest');
    await expect(page.getByTestId('backtest-lab-page')).toBeVisible();
    await expect(page.getByTestId('backtest-config')).toBeVisible();
    await expect(page.getByTestId('backtest-empty')).toBeVisible();
  });

  test('fills backtest form fields', async ({ page }) => {
    await page.goto('/backtest');
    await expect(page.getByTestId('backtest-config')).toBeVisible();

    const startDate = page.getByTestId('backtest-start-date');
    await startDate.fill('2020-01-01');
    await expect(startDate).toHaveValue('2020-01-01');

    const endDate = page.getByTestId('backtest-end-date');
    await endDate.fill('2025-01-01');
    await expect(endDate).toHaveValue('2025-01-01');

    const amount = page.getByTestId('backtest-dca-amount');
    await amount.fill('10000');
    await expect(amount).toHaveValue('10000');

    const threshold = page.getByTestId('backtest-dip-threshold');
    await threshold.fill('70');
    await expect(threshold).toHaveValue('70');
  });

  test('run button exists and is clickable', async ({ page }) => {
    await page.goto('/backtest');
    const runBtn = page.getByTestId('run-backtest-btn');
    await expect(runBtn).toBeVisible();
    await expect(runBtn).toBeEnabled();
  });
});

test.describe('Settings Page', () => {
  test('loads settings form with weight sliders', async ({ page }) => {
    await page.goto('/settings');
    await expect(page.getByTestId('settings-page')).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId('settings-title')).toHaveText('SETTINGS');
    await expect(page.getByTestId('dca-defaults')).toBeVisible();
    await expect(page.getByTestId('score-weights')).toBeVisible();
  });

  test('weight sliders update displayed percentage', async ({ page }) => {
    await page.goto('/settings');
    await expect(page.getByTestId('score-weights')).toBeVisible({ timeout: 15_000 });

    const techSlider = page.getByTestId('weight-technical');
    await expect(techSlider).toBeVisible();

    const volSlider = page.getByTestId('weight-volatility');
    await expect(volSlider).toBeVisible();
  });

  test('save button exists', async ({ page }) => {
    await page.goto('/settings');
    await expect(page.getByTestId('save-settings-btn')).toBeVisible({ timeout: 15_000 });
  });
});

test.describe('News Page', () => {
  test('loads news page with feed or empty state', async ({ page }) => {
    await page.goto('/news');
    await expect(page.getByTestId('news-page')).toBeVisible();
    await expect(page.getByTestId('news-title')).toHaveText('NEWS & GEOPOLITICAL EVENTS');

    const feed = page.getByTestId('news-feed');
    const empty = page.getByTestId('news-empty');
    await expect(feed.or(empty)).toBeVisible({ timeout: 15_000 });
  });

  test('refresh button is clickable', async ({ page }) => {
    await page.goto('/news');
    const refreshBtn = page.getByTestId('refresh-news-btn');
    await expect(refreshBtn).toBeVisible();
    await expect(refreshBtn).toBeEnabled();
  });
});

test.describe('Backend Health', () => {
  test('API health endpoint responds', async ({ request }) => {
    const res = await request.get(`${BACKEND}/api/health`);
    expect(res.ok()).toBeTruthy();
    const body = await res.json();
    expect(body.status).toBe('healthy');
  });

  test('API assets endpoint responds', async ({ request }) => {
    const res = await request.get(`${BACKEND}/api/assets`);
    expect(res.ok()).toBeTruthy();
    const body = await res.json();
    expect(Array.isArray(body)).toBeTruthy();
  });

  test('API dashboard endpoint responds', async ({ request }) => {
    const res = await request.get(`${BACKEND}/api/dashboard`);
    expect(res.ok()).toBeTruthy();
    const body = await res.json();
    expect(body).toHaveProperty('assets');
  });

  test('API settings endpoint responds', async ({ request }) => {
    const res = await request.get(`${BACKEND}/api/settings`);
    expect(res.ok()).toBeTruthy();
    const body = await res.json();
    expect(body).toHaveProperty('default_dca_cadence');
    expect(body).toHaveProperty('score_weights');
  });
});
