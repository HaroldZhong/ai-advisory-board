import { expect, test } from '@playwright/test';

const API_BASE = 'http://localhost:8001';
const CONVERSATION_ID = 'e2e-private-conversation';
const MODEL_GLM = 'z-ai/glm-5.1';
const MODEL_QWEN = 'qwen/qwen3.5-35b-a3b';
const MODEL_CLAUDE = 'anthropic/claude-opus-4.7';
const MODEL_CLAUDE_HAIKU = 'anthropic/claude-haiku-4.5';
const MODEL_CLAUDE_SONNET = 'anthropic/claude-sonnet-5';
const MODEL_GPT = 'openai/gpt-5.5-pro';
const MODEL_GPT_ZDR = 'openai/gpt-5.4';
const SLOW_KEY = 'sk-or-v1-slow-probe-key';

function json(body, status = 200) {
  return {
    status,
    contentType: 'application/json',
    headers: {
      'access-control-allow-origin': '*',
      'access-control-allow-headers': '*',
      'access-control-allow-methods': 'GET,POST,PUT,DELETE,OPTIONS',
    },
    body: JSON.stringify(body),
  };
}

function sse(events) {
  return {
    status: 200,
    contentType: 'text/event-stream',
    headers: {
      'access-control-allow-origin': '*',
      'cache-control': 'no-cache',
    },
    body: events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(''),
  };
}

function model(id, name, { type = 'both', supportsZdr = true, input = 1, output = 2 } = {}) {
  return {
    id,
    name,
    type,
    pricing: { input, output },
    capabilities: ['reasoning', 'generalist'],
    supports_zdr: supportsZdr,
    supports_reasoning: true,
    available: true,
  };
}

const modelsPayload = {
  models: [
    model(MODEL_CLAUDE_HAIKU, 'Anthropic: Claude Haiku 4.5 Extended Reasoning Preview', { input: 1, output: 5 }),
    model(MODEL_CLAUDE_SONNET, 'Anthropic: Claude Sonnet 5 Deep Research Preview', { input: 3, output: 15 }),
    model(MODEL_CLAUDE, 'Anthropic: Claude Opus 4.8 Chairman Default Preview', { input: 5, output: 25 }),
    model(MODEL_GLM, 'Z.ai: GLM 5.1', { input: 1.05, output: 3.5 }),
    model(MODEL_QWEN, 'Qwen: Qwen3.5-35B-A3B', { input: 0.6, output: 1.2 }),
    model(MODEL_GPT, 'OpenAI: GPT-5.5 Pro', { supportsZdr: false, input: 30, output: 120 }),
    model(MODEL_GPT_ZDR, 'OpenAI: GPT-5.4', { input: 2, output: 8 }),
  ],
  defaults: {
    chairman: MODEL_CLAUDE,
    council: [MODEL_GLM, MODEL_QWEN, MODEL_GPT_ZDR],
  },
  presets: [
    {
      id: 'balanced',
      label: 'Balanced',
      description: 'Diverse panel across major labs. The default for most questions.',
      chairman_model: MODEL_CLAUDE,
      council_models: [MODEL_GLM, MODEL_QWEN, MODEL_GPT_ZDR],
      requires_zdr: false,
      default_reasoning_effort: 'medium',
    },
    {
      id: 'research',
      label: 'Research',
      description: 'Frontier-heavy panel for deeper synthesis.',
      chairman_model: MODEL_GPT,
      council_models: [MODEL_GPT, MODEL_GLM],
      requires_zdr: false,
      default_reasoning_effort: 'high',
    },
    {
      id: 'private',
      label: 'Private',
      description: 'ZDR-only panel for sensitive work.',
      chairman_model: MODEL_CLAUDE,
      council_models: [MODEL_GLM, MODEL_QWEN, MODEL_GPT_ZDR],
      requires_zdr: true,
      default_reasoning_effort: 'medium',
    },
  ],
};

function createConversation(body) {
  return {
    id: CONVERSATION_ID,
    created_at: '2026-05-03T00:00:00Z',
    title: body.topic || 'New Conversation',
    total_cost: 0,
    messages: [],
    metadata: {
      preset_id: body.preset_id,
      zdr_enabled: body.zdr_enabled,
      thinking_effort: 'medium',
      chairman_model: MODEL_CLAUDE,
      council_models: [MODEL_GLM, MODEL_QWEN, MODEL_GPT_ZDR],
      default_mode: body.default_mode ?? undefined,
    },
    session_policy: {
      budget_usd: body.budget_usd,
      notify_thresholds: [0.75, 0.85, 1],
      mode: 'auto',
      // echo the requested overage choice (v1.3.0 D3 default is allow-overage)
      allow_overage: body.budget_allow_overage ?? true,
    },
    session_usage: {
      spent_usd: 0,
      messages: 0,
      last_warning_level: null,
    },
    budget_spent_pct: 0,
  };
}

async function installMockApi(page) {
  let hasApiKey = false;
  let conversation = null;
  const requests = {
    setup: null,
    createConversation: null,
    stream: null,
  };

  await page.route(`${API_BASE}/api/config/status`, (route) => {
    route.fulfill(json({ has_api_key: hasApiKey }));
  });

  await page.route(`${API_BASE}/api/config/setup`, async (route) => {
    requests.setup = await route.request().postDataJSON();
    hasApiKey = true;
    route.fulfill(json({ success: true, has_api_key: true }));
  });

  await page.route(`${API_BASE}/api/config/connectivity`, async (route) => {
    const body = route.request().postDataJSON?.() ?? null;
    const delayMs = body?.api_key === SLOW_KEY ? 300 : 0;
    if (delayMs) await new Promise((resolve) => setTimeout(resolve, delayMs));
    route.fulfill(json({ reachable: true, key_valid: true, error_kind: null, detail: '' }));
  });

  await page.route(`${API_BASE}/api/models`, (route) => {
    route.fulfill(json(modelsPayload));
  });

  await page.route(`${API_BASE}/api/folders`, (route) => {
    route.fulfill(json([]));
  });

  await page.route(`${API_BASE}/api/conversations`, async (route) => {
    if (route.request().method() === 'GET') {
      const list = conversation
        ? [{
            id: conversation.id,
            title: conversation.title,
            created_at: conversation.created_at,
            message_count: conversation.messages.length,
          }]
        : [];
      await route.fulfill(json(list));
      return;
    }

    if (route.request().method() === 'POST') {
      requests.createConversation = await route.request().postDataJSON();
      conversation = createConversation(requests.createConversation);
      await route.fulfill(json(conversation));
      return;
    }

    await route.fulfill(json({ detail: 'Unsupported conversation method' }, 405));
  });

  await page.route(new RegExp(`${API_BASE}/api/conversations/${CONVERSATION_ID}/estimate`), async (route) => {
    // v1.3.0 D3 soft seatbelt: an approximate pre-send estimate (POST), mode-aware
    // like the real endpoint -- a full council turn is "large" (warn), an ordinary
    // chat turn is not (dispatches uninterrupted). Mode is in the POST body.
    const isCouncil = (route.request().postDataJSON?.()?.mode ?? 'council') === 'council';
    await route.fulfill(json(
      isCouncil
        ? { predicted_cost: 0.2038, approximate: true, threshold: 0.15, is_large: true }
        : { predicted_cost: 0.045, approximate: true, threshold: 0.15, is_large: false },
    ));
  });

  await page.route(new RegExp(`${API_BASE}/api/conversations/${CONVERSATION_ID}$`), async (route) => {
    if (route.request().method() === 'GET') {
      await route.fulfill(json(conversation));
      return;
    }

    await route.fulfill(json(conversation));
  });

  await page.route(`${API_BASE}/api/conversations/${CONVERSATION_ID}/message/stream`, async (route) => {
    requests.stream = await route.request().postDataJSON();

    conversation.messages = [
      ...conversation.messages,
      { role: 'user', content: requests.stream.content },
      {
        role: 'assistant',
        stage1: [{
          model: MODEL_GLM,
          response: 'The subtraction leaves five.',
          reasoning: 'Eight minus three removes three units from eight.',
        }],
        stage2: [{
          model: MODEL_QWEN,
          ranking: 'A > B',
          parsed_ranking: ['A'],
          reasoning: 'The first answer is concise and correct.',
        }],
        stage3: {
          model: MODEL_CLAUDE,
          response: 'The answer is 5.',
          reasoning: 'Subtracting 3 from 8 leaves 5.',
        },
        metadata: {
          label_to_model: { A: MODEL_GLM },
          aggregate_rankings: [{ model: MODEL_GLM, average_rank: 1, rankings_count: 1 }],
        },
        running_cost: 0.000356,
      },
    ];
    conversation.total_cost = 0.000356;
    conversation.session_usage = {
      spent_usd: 0.000356,
      messages: 1,
      last_warning_level: null,
    };
    conversation.budget_spent_pct = 0.000178;

    await route.fulfill(sse([
      { type: 'stage1_start' },
      {
        type: 'reasoning_delta',
        data: {
          scope: 'council',
          stage: 'stage1',
          model: MODEL_GLM,
          index: 0,
          text: 'Eight minus three removes three units from eight.',
        },
      },
      {
        type: 'stage1_complete',
        data: [{
          model: MODEL_GLM,
          response: 'The subtraction leaves five.',
        }],
      },
      { type: 'stage2_start' },
      {
        type: 'stage2_complete',
        data: [{
          model: MODEL_QWEN,
          ranking: 'A > B',
          parsed_ranking: ['A'],
          reasoning: 'The first answer is concise and correct.',
        }],
        metadata: {
          label_to_model: { A: MODEL_GLM },
          aggregate_rankings: [{ model: MODEL_GLM, average_rank: 1, rankings_count: 1 }],
        },
      },
      { type: 'stage3_start' },
      {
        type: 'reasoning_delta',
        data: {
          scope: 'council',
          stage: 'stage3',
          model: MODEL_CLAUDE,
          text: 'Subtracting 3 from 8 leaves 5.',
        },
      },
      {
        type: 'stage3_complete',
        data: {
          model: MODEL_CLAUDE,
          response: 'The answer is 5.',
          reasoning_tokens: 1200,  // B5/E3 §3d honest actuals
        },
      },
      {
        type: 'complete',
        data: {
          turn_cost: 0.000356,
          total_cost: 0.000356,
          session_usage: conversation.session_usage,
          budget_spent_pct: conversation.budget_spent_pct,
        },
      },
    ]));
  });

  return requests;
}

test.beforeEach(async ({ page }) => {
  // Never fall through to a real backend when a mock route is missing.
  await page.route(`${API_BASE}/**`, (route) => route.abort('blockedbyclient'));
  await page.addInitScript(() => {
    window.localStorage.clear();
  });
});

async function completeSetupToNewConversation(page) {
  const requests = await installMockApi(page);
  await page.setViewportSize({ width: 1024, height: 900 });
  await page.goto('/app');

  await page.getByLabel('OpenRouter API key').fill('sk-or-v1-layout-key');
  await page.getByRole('button', { name: 'Test connection' }).click();
  await expect(page.getByText('Connected to OpenRouter.')).toBeVisible();
  await page.getByRole('button', { name: /Continue/ }).click();
  await page.getByText('Private routing by default').click();
  await page.getByRole('button', { name: /Continue/ }).click();
  await page.getByRole('button', { name: /Finish/ }).click();
  await expect(page.getByRole('heading', { name: 'New conversation' })).toBeVisible();
  return requests;
}

test('desktop model selector recovers from its first failed load without reopening', async ({ page }) => {
  await completeSetupToNewConversation(page);
  await expect(page.getByRole('button', { name: 'Start conversation' })).toBeEnabled();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  let fail = true;
  await page.route(`${API_BASE}/api/models**`, route => route.fulfill(json(modelsPayload, fail ? 503 : 200)));
  await page.getByRole('button', { name: 'New chat', exact: true }).click();
  await expect(page.getByText('Failed to load models', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start conversation' })).toBeDisabled();
  fail = false;
  await page.getByRole('button', { name: 'Refresh models' }).click();
  await expect(page.getByText('Failed to load models', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Start conversation' })).toBeEnabled();
  await expect(page.getByRole('button', { name: /Claude Opus 4.8 Chairman Default Preview.*\/M in/ })).toHaveAttribute('aria-pressed', 'true');
});

test('desktop council selection can explicitly replace a model removed by its provider', async ({ page }) => {
  const requests = await completeSetupToNewConversation(page);
  await expect(page.getByRole('button', { name: 'Start conversation' })).toBeEnabled();
  await page.getByRole('button', { name: 'Council', exact: true }).click();
  await page.getByRole('button', { name: 'Custom', exact: true }).click();
  const selected = page.getByRole('group', { name: 'Selected council models', exact: true });
  await expect(selected.getByRole('button', { name: `Remove ${MODEL_GLM}`, exact: true })).toBeVisible();
  await page.route(`${API_BASE}/api/models**`, route => route.fulfill(json({
    ...modelsPayload, models: modelsPayload.models.filter(entry => entry.id !== MODEL_GLM),
  })));
  await page.getByRole('button', { name: 'Refresh models' }).click();
  await expect(selected.getByText('Needs replacement', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start conversation' })).toBeDisabled();
  await selected.getByRole('button', { name: `Remove ${MODEL_GLM}`, exact: true }).click();
  await expect(page.getByRole('button', { name: 'Start conversation' })).toBeDisabled();
  await page.getByRole('button', { name: /Claude Haiku 4.5 Extended Reasoning Preview.*\/M in/ }).click();
  await expect(page.getByRole('button', { name: 'Start conversation' })).toBeEnabled();
  await page.getByRole('button', { name: 'Start conversation' }).click();
  await expect.poll(() => requests.createConversation).not.toBeNull();
  expect(requests.createConversation.council_models).toEqual([MODEL_QWEN, MODEL_GPT_ZDR, MODEL_CLAUDE_HAIKU]);
});

test('desktop model refresh preserves choices, discovers models and reports removal/failure', async ({ page }) => {
  await completeSetupToNewConversation(page);
  await page.setViewportSize({ width: 1280, height: 900 });
  let payload = {
    ...modelsPayload,
    models: [...modelsPayload.models, model('new/fresh', 'New: Fresh Model', { input: null, output: null })],
    catalog: { last_fetched: 1789600000, stale: false },
  };
  let fail = false;
  await page.route(`${API_BASE}/api/models**`, route => route.fulfill(json(payload, fail ? 503 : 200)));
  await page.getByRole('button', { name: 'Refresh models' }).click();
  await expect(page.getByRole('button', { name: /Claude Opus 4.8 Chairman Default Preview.*\/M in/ })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('textbox', { name: 'Search models' }).fill('new/fresh');
  const fresh = page.getByRole('button', { name: /Fresh Model.*Price not reported/ });
  await expect(fresh).toBeVisible();
  await fresh.click();
  await expect(page.getByText('n/a est.', { exact: true })).toBeVisible();
  payload = { ...payload, models: payload.models.map(entry => entry.id === 'new/fresh' ? { ...entry, pricing: { input: 9, output: 10 } } : entry) };
  await page.getByRole('button', { name: 'Refresh models' }).click();
  await expect(page.getByRole('button', { name: /Fresh Model.*\$9\/M in/ })).toHaveAttribute('aria-pressed', 'true');
  payload = { ...payload, models: modelsPayload.models };
  await page.getByRole('button', { name: 'Refresh models' }).click();
  await expect(page.getByText(/A selected model is unavailable/)).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start conversation' })).toBeDisabled();
  await expect(page.getByText('new/fresh', { exact: true })).toBeVisible();
  fail = true;
  await page.getByRole('button', { name: 'Refresh models' }).click();
  await expect(page.getByText('Refresh failed. Keeping the previous catalog.', { exact: true })).toBeVisible();
  await expect(page.getByText('new/fresh', { exact: true })).toBeVisible();
});

test('active desktop discovers catalog changes without clicking refresh', async ({ page }) => {
  await page.clock.install();
  await completeSetupToNewConversation(page);
  await expect(page.getByRole('button', { name: 'Start conversation' })).toBeEnabled();
  await page.route(`${API_BASE}/api/models**`, route => route.fulfill(json({
    ...modelsPayload, models: [...modelsPayload.models, model('new/auto', 'New: Automatic Discovery')],
    catalog: { last_fetched: 1789600000, stale: false },
  })));
  await page.clock.fastForward(60_000);
  await page.getByRole('textbox', { name: 'Search models' }).fill('new/auto');
  await expect(page.getByRole('button', { name: /Automatic Discovery.*\/M in/ })).toBeVisible();
});

async function expectVisibleModelGridsFit(page, label) {
  const metrics = await page.locator('[role="dialog"]').evaluate((dialog) => {
    const dialogRect = dialog.getBoundingClientRect();
    const scrollWrappers = [...dialog.querySelectorAll('[data-radix-scroll-area-viewport]')]
      .map((viewport) => {
        const content = viewport.firstElementChild;
        if (!content?.querySelector?.('[role="button"][aria-pressed]')) return null;
        const viewportRect = viewport.getBoundingClientRect();
        const contentRect = content.getBoundingClientRect();
        return {
          display: window.getComputedStyle(content).display,
          viewportClientWidth: viewport.clientWidth,
          contentScrollWidth: content.scrollWidth,
          contentRight: contentRect.right,
          viewportRight: viewportRect.right,
          hasModelCards: [...content.querySelectorAll('[role="button"][aria-pressed]')]
            .some((card) => card.textContent?.includes('/M in /')),
        };
      })
      .filter((wrapper) => wrapper?.hasModelCards);
    const grids = [...dialog.querySelectorAll('.grid')]
      .map((grid) => {
        const gridRect = grid.getBoundingClientRect();
        const cards = [...grid.querySelectorAll('[role="button"][aria-pressed]')]
          .map((card) => {
            const rect = card.getBoundingClientRect();
            return {
              text: card.textContent?.replace(/\s+/g, ' ').trim(),
              left: rect.left,
              right: rect.right,
              width: rect.width,
              visible: rect.width > 0 && rect.height > 0,
            };
          })
          .filter((card) => card.visible && card.text?.includes('/M in /'));

        return {
          gridClientWidth: grid.clientWidth,
          gridScrollWidth: grid.scrollWidth,
          gridRight: gridRect.right,
          cards,
        };
      })
      .filter((grid) => grid.cards.length >= 3);

    return {
      dialogRight: dialogRect.right,
      scrollWrappers,
      grids,
    };
  });

  expect(metrics.scrollWrappers.length, `${label} should render model cards inside a ScrollArea`).toBeGreaterThan(0);
  for (const wrapper of metrics.scrollWrappers) {
    expect(wrapper.display, `${label} ScrollArea content wrapper should let grids resolve against the viewport`).toBe('block');
    expect(wrapper.contentScrollWidth, `${label} ScrollArea content should not horizontally overflow`).toBeLessThanOrEqual(wrapper.viewportClientWidth + 1);
    expect(wrapper.contentRight, `${label} ScrollArea content clipped at viewport right edge`).toBeLessThanOrEqual(wrapper.viewportRight + 1);
  }

  expect(metrics.grids.length, `${label} should render a visible 3-card model grid`).toBeGreaterThan(0);
  for (const grid of metrics.grids) {
    expect(grid.gridScrollWidth, `${label} model grid should not horizontally overflow`).toBeLessThanOrEqual(grid.gridClientWidth + 1);
    for (const card of grid.cards) {
      expect(card.right, `${label} model card clipped at modal right edge: ${card.text}`).toBeLessThanOrEqual(metrics.dialogRight + 1);
    }
  }
}

test('first-run setup creates a private preset conversation and renders streamed reasoning', async ({ page }) => {
  const requests = await installMockApi(page);

  await page.goto('/app');

  await expect(page.getByRole('heading', { name: 'Set Up AI Advisory Board' })).toBeVisible();
  await expect(page.getByRole('button', { name: /Continue/ })).toBeDisabled();

  await page.getByLabel('OpenRouter API key').fill('sk-or-v1-launch-hardening-key');
  await page.getByRole('button', { name: 'Test connection' }).click();
  await expect(page.getByText('Connected to OpenRouter.')).toBeVisible();

  // Editing the key after a successful test must clear the stale result.
  await page.getByLabel('OpenRouter API key').fill('sk-or-v1-launch-hardening-key-edited');
  await expect(page.getByText('Connected to OpenRouter.')).not.toBeVisible();

  await page.getByRole('button', { name: 'Test connection' }).click();
  await expect(page.getByText('Connected to OpenRouter.')).toBeVisible();
  await page.getByRole('button', { name: /Continue/ }).click();
  await page.getByText('Private routing by default').click();
  await page.getByRole('button', { name: /Continue/ }).click();
  await page.getByRole('button', { name: /Finish/ }).click();

  await expect(page.getByRole('heading', { name: 'New conversation' })).toBeVisible();
  // The dialog defaults to Chat mode (P3-T3); switch to Council to reach the
  // preset picker this test exercises.
  await page.getByRole('button', { name: 'Council' }).click();
  await expect(page.getByRole('button', { name: /Research/ })).toContainText('hidden by ZDR');
  await page.getByRole('button', { name: /Private ZDR-only panel/ }).click();
  await page.getByRole('button', { name: 'Start conversation' }).click();

  await expect(page).toHaveURL(new RegExp(`/c/${CONVERSATION_ID}$`));
  expect(requests.setup).toEqual({ api_key: 'sk-or-v1-launch-hardening-key-edited' });
  expect(requests.createConversation).toMatchObject({
    preset_id: 'private',
    zdr_enabled: true,
    budget_usd: 2,
    // v1.3.0 D3: new conversations allow overage by default (warn, don't block).
    budget_allow_overage: true,
  });

  await expect(page.getByText('Private').first()).toBeVisible();
  await expect(page.getByText('ZDR enforced')).toBeVisible();

  await page.getByRole('textbox', { name: /Ask your question/ }).fill('What is 8 minus 3?');
  await page.getByRole('button', { name: 'Send message' }).click();

  // v1.3.0 D3 (§5.1): the automatic first council turn now warns with an approximate
  // pre-send estimate (the main council-default path, not just manual "Ask the
  // council"). Confirm to dispatch.
  await expect(page.getByRole('alert')).toContainText('Est. ~$0.20 (approximate)');
  await page.getByRole('button', { name: 'Confirm', exact: true }).click();

  await expect(page.getByText('Final Council Answer')).toBeVisible();
  // B5/E3 §3d: the chairman's honest post-turn reasoning actuals render from the count
  await expect(page.getByText('reasoning: 1.2k tokens')).toBeVisible();
  await expect(page.getByText('Reasoning complete').first()).toBeVisible();
  await expect(page.getByText('The answer is 5.')).toBeVisible();
  await expect(page.getByText('Recorded cost:')).toBeVisible();
  await expect(page.getByText('$0.000356')).toBeVisible();
  await expect(page.getByText('Session cost')).toBeVisible();

  expect(requests.stream).toMatchObject({
    content: 'What is 8 minus 3?',
    // Routing is backend-owned (P3-T1): the wire always carries "auto".
    mode: 'auto',
    zdr_enabled: true,
  });
});

test('new-conversation model grids fit inside the modal on wide screens', async ({ page }) => {
  await completeSetupToNewConversation(page);

  await expectVisibleModelGridsFit(page, 'Chat');

  await page.getByRole('button', { name: 'Council' }).click();
  await page.getByRole('button', { name: 'Custom' }).click();
  await expectVisibleModelGridsFit(page, 'Council');
});

test('first-run setup can be dismissed and lands on the landing page', async ({ page }) => {
  await installMockApi(page);

  await page.goto('/app');

  await expect(page.getByRole('heading', { name: 'Set Up AI Advisory Board' })).toBeVisible();

  await page.keyboard.press('Escape');

  await expect(page.getByRole('heading', { name: 'Set Up AI Advisory Board' })).not.toBeVisible();
  await expect(page).toHaveURL('/');
});

test('editing the key mid-probe discards the stale connection result', async ({ page }) => {
  await installMockApi(page);

  await page.goto('/app');

  await page.getByLabel('OpenRouter API key').fill(SLOW_KEY);
  await page.getByRole('button', { name: 'Test connection' }).click();

  // Edit the key while the (deliberately slow) probe for SLOW_KEY is still in flight.
  await page.getByLabel('OpenRouter API key').fill('sk-or-v1-edited-mid-flight');

  // Give the slow probe time to resolve; its result must be discarded, not shown.
  await page.waitForTimeout(500);
  await expect(page.getByText('Connected to OpenRouter.')).not.toBeVisible();
});

test('D3 soft seatbelt: the council confirm surfaces an approximate cost and is dismissible', async ({ page }) => {
  // v1.3.0 D3 (§5.1): before an expensive (council) send, warn with an APPROXIMATE
  // pre-send estimate -- never block. Dismissible so the user can proceed.
  await completeSetupToNewConversation(page);
  await page.getByRole('button', { name: 'Start conversation' }).click();
  await expect(page).toHaveURL(new RegExp(`/c/${CONVERSATION_ID}$`));

  // A chat-default conversation offers "Ask the council" on the first turn.
  await page.getByRole('button', { name: 'Ask the council' }).click();
  await page.getByRole('textbox', { name: /Ask your question/ }).fill('Weigh the trade-offs.');
  await page.getByRole('button', { name: 'Send message' }).click();

  // The confirm appears with the approximate estimate (mocked is_large council turn).
  const confirm = page.getByRole('alert');
  await expect(confirm).toContainText('Est. ~$0.20 (approximate)');
  await expect(confirm).toContainText('costs more');

  // Dismissible: Cancel closes the confirm and no turn is dispatched.
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('D3 soft seatbelt: a large predicted CHAT turn also warns, not only council', async ({ page }) => {
  // v1.3.0 D3 (§5.1): "large predicted spend" is mode-agnostic -- an expensive chat
  // turn (pricey chairman) must warn too, not only a council fan-out (Codex #110 R2).
  await completeSetupToNewConversation(page);
  // Override the estimate so a plain chat turn reads as large.
  await page.route(new RegExp(`${API_BASE}/api/conversations/${CONVERSATION_ID}/estimate`), async (route) => {
    await route.fulfill(json({ predicted_cost: 0.3, approximate: true, threshold: 0.15, is_large: true }));
  });
  await page.getByRole('button', { name: 'Start conversation' }).click();
  await expect(page).toHaveURL(new RegExp(`/c/${CONVERSATION_ID}$`));

  // Plain chat send (council NOT armed) whose estimate is large must still warn.
  await page.getByRole('textbox', { name: /Ask your question/ }).fill('A large chat turn.');
  await page.getByRole('button', { name: 'Send message' }).click();

  const confirm = page.getByRole('alert');
  await expect(confirm).toContainText('larger-than-usual turn');
  await expect(confirm).toContainText('Est. ~$0.30 (approximate)');
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('reopening the budget dialog re-seeds the hard-cap toggle from the saved policy', async ({ page }) => {
  // v1.3.0 D3 regression guard: the budget dialog stays mounted across opens, so a
  // stale/cancelled local toggle must never survive to overwrite the saved policy.
  await completeSetupToNewConversation(page);

  await page.getByRole('button', { name: 'Council' }).click();
  await page.getByRole('button', { name: /Private ZDR-only panel/ }).click();
  await page.getByRole('button', { name: 'Start conversation' }).click();
  await expect(page).toHaveURL(new RegExp(`/c/${CONVERSATION_ID}$`));

  const openBudget = page.getByRole('button', { name: 'Open session budget settings' });
  const hardCap = page.getByRole('checkbox', { name: 'Enforce hard budget cap' });

  // New conversations persist allow_overage=true (D3 default) -> hard cap OFF.
  await openBudget.click();
  await expect(hardCap).not.toBeChecked();

  // D2: truthful meter -- no invented "~N messages" estimates; honest cap copy;
  // the alert tiers are rendered from the served notify_thresholds ([0.75,0.85,1]).
  await expect(page.getByText('Standard cap')).toBeVisible();
  await expect(page.getByText(/~\d+-\d+ messages/)).toHaveCount(0);
  await expect(page.getByText(/alerts at 75%, 85%, and 100% of your budget/)).toBeVisible();

  // Toggle the hard cap ON, then CANCEL without saving.
  await hardCap.check();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();

  // Reopening must reflect the PERSISTED policy again, not the cancelled toggle.
  // Pre-fix the dialog kept allowOverage=false (checkbox stayed checked) and a later
  // "Set Budget" would silently strip the saved policy (409 hard cap -> warn-only).
  await openBudget.click();
  await expect(hardCap).not.toBeChecked();
});

const materialSnapshot = {
  schema_version: 1, content_hash: 'demo-content',
  sources: [{ source_id: 'att_demo1', version_id: 'v1', alias: 'S1', title: '研究资料.pdf', status: 'partial', included: [1], omitted: [2] }],
  citations: { '[S1.1]': { source_id: 'att_demo1', version_id: 'v1', ordinal: 1, page: 2, start: 9, end: 19, text: '项目预算为100万元。' } },
};

async function openResearchConversation(page) {
  const requests = await installMockApi(page);
  const conversation = createConversation({ default_mode: 'chat', budget_usd: 2 });
  conversation.title = 'Research sources';
  conversation.messages = [
    { role: 'user', content: '项目预算？', attachment_ids: ['att_demo1'], attachments: [{ attachment_id: 'att_demo1', filename: '研究资料.pdf', status: 'partial' }] },
    { role: 'assistant', content: '项目预算为100万元 [S1.1]。未核验 [S9.9]。', metadata: {
      schema_version: 1, run_id: 'saved-run', generation_state: 'complete', persistence_state: 'saved', memory_state: 'skipped',
      evidence_snapshot: materialSnapshot, citation_checks: { items: [{ token: '[S1.1]', status: 'valid_locator' }, { token: '[S9.9]', status: 'invalid_id' }] },
    } },
  ];
  await page.route(`${API_BASE}/api/config/status`, (route) => route.fulfill(json({ has_api_key: true })));
  await page.route(`${API_BASE}/api/conversations`, (route) => route.fulfill(json([conversation])));
  await page.route(`${API_BASE}/api/conversations/${CONVERSATION_ID}`, (route) => route.fulfill(json(conversation)));
  await page.route(`${API_BASE}/api/conversations/${CONVERSATION_ID}/estimate`, async (route) => {
    requests.estimate = route.request().postDataJSON();
    await route.fulfill(json({ predicted_cost: 0.01, approximate: true, is_large: false }));
  });
  await page.route(`${API_BASE}/api/conversations/${CONVERSATION_ID}/message/stream`, async (route) => {
    requests.stream = route.request().postDataJSON();
    await route.fulfill(sse([
      { type: 'chat_response', data: { content: 'Done' } },
      { type: 'complete', data: { turn_cost: 0, total_cost: 0, session_usage: {}, budget_spent_pct: 0 } },
    ]));
  });
  await page.goto(`/c/${CONVERSATION_ID}`);
  await expect(page.getByText('项目预算为100万元', { exact: false }).first()).toBeVisible();
  return { requests, conversation };
}

for (const width of [1280, 390]) {
  test(`material citations open their saved source at ${width}px with keyboard dismissal`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await openResearchConversation(page);
    const citation = page.getByRole('button', { name: '[S1.1]', exact: true });
    await citation.focus();
    await page.keyboard.press('Enter');
    const panel = width < 1024 ? page.getByRole('dialog') : page.getByRole('complementary', { name: 'Materials' });
    await expect(panel).toBeVisible();
    await expect(panel.getByText('[S1.1] · Page 2 · Chunk 1', { exact: false })).toBeVisible();
    await expect(panel.getByText('partial · 1 chunks read · 1 unread')).toBeVisible();
    if (width >= 1024) {
      const content = await page.locator('[data-chat-column]').boundingBox();
      const rail = await panel.boundingBox();
      expect(content.x + content.width).toBeLessThanOrEqual(rail.x + 1);
    }
    await expect(page.getByRole('button', { name: '[S9.9]', exact: true })).toHaveCount(0);
    await page.keyboard.press('Escape');
    await expect(panel).toHaveCount(0);
    await expect(citation).toBeFocused();
    const materialsToggle = page.getByRole('button', { name: 'Toggle materials sidebar' });
    await materialsToggle.click();
    await panel.getByRole('button', { name: 'Materials', exact: true }).focus();
    await page.keyboard.press('Escape');
    await expect(materialsToggle).toBeFocused();
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
    expect(overflow).toBe(false);
  });
}

test('empty material scope is identical in estimate and send', async ({ page }) => {
  const { requests } = await openResearchConversation(page);
  if (!(await page.getByRole('button', { name: 'Use none', exact: true }).isVisible())) await page.getByRole('button', { name: 'Toggle materials sidebar' }).click();
  await page.getByRole('button', { name: 'Use none', exact: true }).click();
  await page.getByRole('textbox', { name: /Ask your question/ }).fill('Do not use materials this turn');
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await expect.poll(() => requests.stream).not.toBeNull();
  expect(requests.estimate.evidence_source_ids).toEqual([]);
  expect(requests.stream.evidence_source_ids).toEqual([]);
  expect(requests.estimate.attachment_ids).toEqual(requests.stream.attachment_ids);
});

for (const scenario of ['waiting', 'partial', 'eof']) {
  test(`interrupted ${scenario} response keeps a draft across navigation without claiming it saved`, async ({ page }) => {
    await openResearchConversation(page);
    await page.evaluate((scenario) => {
      const original = window.fetch.bind(window);
      window.fetch = async (url, options) => {
        if (!String(url).endsWith('/message/stream')) return original(url, options);
        const encoder = new TextEncoder();
        const events = [
          { type: 'turn_state', data: { run_id: 'pending-run', generation: 'pending', persistence: 'pending', memory: 'pending' }, metadata: { run_id: 'pending-run' } },
          { type: 'chat_start' },
        ];
        if (scenario !== 'waiting') events.push({ type: 'content_delta', data: { stage: 'chat', text: '未确认的部分正文' } });
        return new Response(new ReadableStream({ start(controller) {
          for (const event of events) controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
          options.signal.addEventListener('abort', () => controller.error(new DOMException('Aborted', 'AbortError')));
          if (scenario === 'eof') controller.close();
        } }), { status: 200, headers: { 'content-type': 'text/event-stream' } });
      };
    }, scenario);
    await page.getByRole('textbox', { name: /Ask your question/ }).fill('Keep this pending request');
    await page.getByRole('button', { name: 'Send message', exact: true }).click();
    if (scenario !== 'eof') {
      if (scenario === 'partial') await expect(page.getByText('未确认的部分正文', { exact: true })).toBeVisible();
      await page.getByRole('button', { name: 'Stop response', exact: true }).click();
      await expect(page.getByText('Stop requested.', { exact: false }).first()).toBeVisible();
      await expect(page.getByText('Response failed', { exact: true })).toHaveCount(0);
    } else {
      await expect(page.getByText('Connection lost.', { exact: false }).first()).toBeVisible();
    }
    await expect(page.getByText('Unconfirmed drafts (1)', { exact: true })).toBeVisible();
    await page.evaluate(() => { history.pushState({}, '', '/app'); window.dispatchEvent(new PopStateEvent('popstate')); });
    await page.getByText('Unconfirmed drafts (1)', { exact: true }).click();
    const draft = page.getByRole('textbox', { name: 'Unconfirmed response text' });
    await expect(draft).toContainText('Keep this pending request');
    if (scenario !== 'waiting') await expect(draft).toContainText('未确认的部分正文');
    await page.getByRole('button', { name: 'Discard this draft', exact: true }).click();
    await expect(draft).toHaveCount(0);
  });
}

test('stop during memory indexing reconciles the saved run without creating a false draft', async ({ page }) => {
  const { conversation } = await openResearchConversation(page);
  await page.evaluate(() => {
    const original = window.fetch.bind(window);
    window.fetch = async (url, options) => {
      if (!String(url).endsWith('/message/stream')) return original(url, options);
      const events = [
        { type: 'chat_start' },
        { type: 'chat_response', data: { content: 'Already saved answer' } },
        { type: 'turn_state', data: { run_id: 'committed-run', generation: 'complete', persistence: 'saved', memory: 'pending' }, metadata: { run_id: 'committed-run', persistence_state: 'saved', memory_state: 'pending' } },
      ];
      return new Response(new ReadableStream({ start(controller) {
        const encoder = new TextEncoder();
        for (const event of events) controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
        options.signal.addEventListener('abort', () => controller.error(new DOMException('Aborted', 'AbortError')));
      } }), { headers: { 'content-type': 'text/event-stream' } });
    };
  });
  await page.getByRole('textbox', { name: /Ask your question/ }).fill('Save before indexing');
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await expect(page.getByText('Already saved answer', { exact: true })).toBeVisible();
  conversation.messages.push({ role: 'user', content: 'Save before indexing' }, {
    role: 'assistant', content: 'Already saved answer', metadata: { schema_version: 1, run_id: 'committed-run', generation_state: 'complete', persistence_state: 'saved', memory_state: 'pending' },
  });
  await page.getByRole('button', { name: 'Stop response', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Send message', exact: true })).toBeVisible();
  await expect(page.getByText('Already saved answer', { exact: true })).toBeVisible();
  await expect(page.getByText('Answer: complete · Saved: saved · Memory: pending', { exact: true })).toBeVisible();
  await expect(page.getByText('Unconfirmed drafts (1)', { exact: true })).toHaveCount(0);
});

test('Chat without a roster configures Council in the same conversation before estimating or sending', async ({ page }) => {
  const { requests, conversation } = await openResearchConversation(page);
  delete conversation.metadata.council_models;
  conversation.metadata.chairman_model = 'local/chat';
  await page.route(`${API_BASE}/api/config/status`, (route) => route.fulfill(json({ has_api_key: true, provider_kind: 'openai-compatible' })));
  let configured;
  await page.route(`${API_BASE}/api/conversations/${CONVERSATION_ID}`, async (route) => {
    if (route.request().method() === 'PUT') {
      configured = route.request().postDataJSON();
      conversation.metadata = { ...conversation.metadata, ...configured };
    }
    await route.fulfill(json(conversation));
  });
  await page.route(`${API_BASE}/api/conversations/${CONVERSATION_ID}/estimate`, async (route) => {
    requests.estimate = route.request().postDataJSON();
    await route.fulfill(json({ predicted_cost: 0.01, approximate: true, is_large: false, council_models: conversation.metadata.council_models, chairman_model: conversation.metadata.chairman_model }));
  });
  await page.reload();
  await page.getByRole('button', { name: 'Ask the council', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Configure council', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Chat', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Use these models', exact: true }).click();
  await expect.poll(() => configured).toBeTruthy();
  expect(configured.council_models.length).toBeGreaterThanOrEqual(3);
  expect(configured.chairman_model).toBe('local/chat');
  expect(requests.stream).toBeNull();
  await page.getByRole('textbox', { name: /Ask your question/ }).fill('Use the configured roster');
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await expect(page.getByText(`Members: ${configured.council_models.join(', ')}`, { exact: false })).toBeVisible();
  await expect(page.getByText('Chairman: local/chat', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: 'Confirm', exact: true }).click();
  await expect.poll(() => requests.stream).not.toBeNull();
  expect(requests.stream.mode).toBe('council');
  await expect(page).toHaveURL(new RegExp(`/c/${CONVERSATION_ID}$`));
});

test('a changed Council roster restores the draft and requires a fresh confirmation', async ({ page }) => {
  const { conversation, requests } = await openResearchConversation(page);
  const original = ['local/a', 'local/b', 'local/c'];
  let roster = original;
  await page.route(`${API_BASE}/api/conversations/${CONVERSATION_ID}/estimate`, async (route) => {
    requests.estimate = route.request().postDataJSON();
    await route.fulfill(json({ predicted_cost: 0.01, approximate: true, is_large: false, council_models: roster, chairman_model: 'local/chair' }));
  });
  let sends = [];
  await page.route(`${API_BASE}/api/conversations/${CONVERSATION_ID}/message/stream`, async (route) => {
    sends.push(route.request().postDataJSON());
    roster = ['local/new-a', 'local/new-b', 'local/new-c'];
    await route.fulfill({ ...json({ detail: 'Council models changed. Review the new estimate and confirm again.' }), status: 412 });
  });
  const prompt = page.getByRole('textbox', { name: /Ask your question/ });
  if (!(await page.getByRole('button', { name: 'Use none', exact: true }).isVisible())) await page.getByRole('button', { name: 'Toggle materials sidebar' }).click();
  await page.getByRole('button', { name: 'Use none', exact: true }).click();
  await page.getByRole('button', { name: 'Ask the council', exact: true }).click();
  await prompt.fill('Keep my draft and selected scope');
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await expect(page.getByText(`Members: ${original.join(', ')}`, { exact: false })).toBeVisible();
  await page.getByRole('button', { name: 'Confirm', exact: true }).click();
  await expect(prompt).toHaveValue('Keep my draft and selected scope');
  await expect(page.getByText('Council models changed. Review the new estimate and confirm again.', { exact: true })).toBeVisible();
  await expect(page.getByRole('dialog')).toHaveCount(0);
  expect(sends[0].expected_council_models).toEqual(original);
  expect(sends[0].expected_chairman_model).toBe('local/chair');
  expect(sends[0].evidence_source_ids).toEqual([]);
  expect(conversation.messages).toHaveLength(2);
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await expect(page.getByText(`Members: ${roster.join(', ')}`, { exact: false })).toBeVisible();
  expect(sends).toHaveLength(1);
  await page.getByRole('button', { name: 'Confirm', exact: true }).click();
  await expect.poll(() => sends.length).toBe(2);
  expect(sends[1].expected_council_models).toEqual(roster);
  expect(sends[1].evidence_source_ids).toEqual([]);
});

test('a late Council precondition error cannot overwrite another conversation draft', async ({ page }) => {
  const { conversation } = await openResearchConversation(page);
  let release;
  const pending = new Promise((resolve) => { release = resolve; });
  let requested = false;
  await page.route(`${API_BASE}/api/conversations/${CONVERSATION_ID}/message/stream`, async (route) => {
    requested = true;
    await pending;
    await route.fulfill({ ...json({ detail: 'Council models changed. Review the new estimate and confirm again.' }), status: 412 });
  });
  const next = { ...conversation, id: 'other-conversation', messages: [] };
  await page.route(`${API_BASE}/api/conversations/other-conversation`, (route) => route.fulfill(json(next)));
  await page.getByRole('button', { name: 'Ask the council', exact: true }).click();
  await page.getByRole('textbox', { name: /Ask your question/ }).fill('Old conversation draft');
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await page.getByRole('button', { name: 'Confirm', exact: true }).click();
  await expect.poll(() => requested).toBe(true);
  await page.evaluate(() => { history.pushState({}, '', '/c/other-conversation'); window.dispatchEvent(new PopStateEvent('popstate')); });
  await expect(page).toHaveURL(/other-conversation$/);
  release();
  const prompt = page.getByRole('textbox', { name: /Ask your question/ });
  await expect(prompt).toBeEnabled();
  await expect(prompt).toHaveValue('');
  await prompt.fill('New conversation draft');
  await expect(page.getByText('Council models changed. Review the new estimate and confirm again.', { exact: true })).toHaveCount(0);
  await expect(prompt).toHaveValue('New conversation draft');
});


for (const width of [1280, 1024, 1440]) {
  test(`materials workspace preserves scope separately from saved sources at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 844 });
    const { requests } = await openResearchConversation(page);
    await expect(page.locator('summary').filter({ hasText: 'Materials for this turn' })).toHaveCount(0);
    const toggle = page.getByRole('button', { name: 'Toggle materials sidebar' });
    if (width < 1024) await toggle.click();
    const panel = width < 1024 ? page.getByRole('dialog') : page.getByRole('complementary', { name: 'Materials' });
    await expect(panel.getByRole('heading', { name: 'Conversation files' })).toBeVisible();
    await expect(panel.getByText('Partially extracted', { exact: true })).toBeVisible();
    await panel.getByRole('checkbox', { name: 'Use 研究资料.pdf' }).uncheck();
    await expect(panel.getByText('The next answer will not use conversation files.')).toBeVisible();
    if (width < 1024) await page.keyboard.press('Escape');
    await page.getByRole('button', { name: '[S1.1]', exact: true }).click();
    await expect(panel.getByText('项目预算为100万元。', { exact: false })).toBeVisible();
    await panel.getByRole('button', { name: 'Materials', exact: true }).click();
    await expect(panel.getByRole('checkbox', { name: 'Use 研究资料.pdf' })).not.toBeChecked();
    await panel.getByRole('button', { name: 'Use all', exact: true }).click();
    if (width < 1024) await page.keyboard.press('Escape');
    const input = page.getByRole('textbox', { name: 'Ask your question', exact: true });
    const inputBox = await input.boundingBox();
    expect(inputBox.y + inputBox.height).toBeLessThanOrEqual(844);
    const column = await page.locator('[data-chat-column]').boundingBox();
    expect(column.width).toBeGreaterThanOrEqual(560);
    // Compact controls leave at least half the desktop window height for reading.
    expect(inputBox.y).toBeGreaterThan(422);
    await input.fill('Use the conversation files again');
    await page.getByRole('button', { name: 'Send message', exact: true }).click();
    await expect.poll(() => requests.stream).not.toBeNull();
    expect(requests.estimate.evidence_source_ids ?? null).toBeNull();
    expect(requests.stream.evidence_source_ids ?? null).toBeNull();
    expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)).toBe(false);
  });
}
