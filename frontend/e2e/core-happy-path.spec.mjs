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
    // v1.3.0 D3 soft seatbelt: an approximate pre-send estimate, mode-aware like the
    // real endpoint -- a full council turn is "large" (warn), an ordinary chat turn
    // is not (dispatches uninterrupted).
    const isCouncil = new URL(route.request().url()).searchParams.get('mode') === 'council';
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
  await page.addInitScript(() => {
    window.localStorage.clear();
  });
});

async function completeSetupToNewConversation(page) {
  await installMockApi(page);
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
}

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
  await page.getByRole('button', { name: 'Confirm' }).click();

  await expect(page.getByText('Final Council Answer')).toBeVisible();
  // B5/E3 §3d: the chairman's honest post-turn reasoning actuals render from the count
  await expect(page.getByText('reasoning: 1.2k tokens')).toBeVisible();
  await expect(page.getByText('Reasoning complete').first()).toBeVisible();
  await expect(page.getByText('The answer is 5.')).toBeVisible();
  await expect(page.getByText('Turn Cost:')).toBeVisible();
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
  await page.getByRole('button', { name: 'Cancel' }).click();
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
  await page.getByRole('button', { name: 'Cancel' }).click();
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
  await page.getByRole('button', { name: 'Cancel' }).click();

  // Reopening must reflect the PERSISTED policy again, not the cancelled toggle.
  // Pre-fix the dialog kept allowOverage=false (checkbox stayed checked) and a later
  // "Set Budget" would silently strip the saved policy (409 hard cap -> warn-only).
  await openBudget.click();
  await expect(hardCap).not.toBeChecked();
});
