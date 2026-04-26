export function modelById(models) {
  return new Map((models || []).map((model) => [model.id, model]));
}

export function isPresetAvailableForZdr(preset, models) {
  if (!preset) return false;
  const byId = modelById(models);
  const ids = [preset.chairman_model, ...(preset.council_models || [])];
  return ids.every((id) => byId.get(id)?.supports_zdr === true);
}

export function getEffectivePresetZdr(preset, zdrEnabled) {
  return Boolean(zdrEnabled || preset?.requires_zdr);
}

export function resolveInitialZdrPreference(settings = {}) {
  return settings.defaultZdrEnabled === true || settings.zdrEnabled === true;
}

export function canStartPresetWithZdr(preset, models, zdrEnabled) {
  const effectiveZdr = getEffectivePresetZdr(preset, zdrEnabled);
  return !effectiveZdr || isPresetAvailableForZdr(preset, models);
}

export function canConfirmModelSelection({
  chairman,
  council,
  selectedPresetAvailable,
  loading,
  error,
  minCouncilSize = 3,
}) {
  return Boolean(chairman)
    && (council || []).length >= minCouncilSize
    && selectedPresetAvailable === true
    && loading !== true
    && !error;
}

export function resolvePresetModels(preset, models, zdrEnabled) {
  if (!preset) return { chairman: null, council: [], hiddenByZdr: 0 };

  const byId = modelById(models);
  const chairman = byId.get(preset.chairman_model) || null;
  const resolvedCouncil = (preset.council_models || [])
    .map((id) => byId.get(id))
    .filter(Boolean);
  const council = resolvedCouncil.filter((model) => !zdrEnabled || model.supports_zdr);

  return {
    chairman,
    council,
    hiddenByZdr: resolvedCouncil.length - council.length,
  };
}

export function estimateSelectionCost({ chairman, council }) {
  const chairmanCost = chairman
    ? Number(chairman.pricing?.input || 0) + Number(chairman.pricing?.output || 0)
    : 0;
  const councilCost = (council || []).reduce((sum, model) => (
    sum
    + Number(model.pricing?.input || 0) * 0.5
    + Number(model.pricing?.output || 0) * 1.5
  ), 0);

  return (chairmanCost + councilCost) / 1000;
}

export function getProvider(modelName = '') {
  const match = modelName.match(/^([^:]+):/);
  if (match) return match[1].trim();

  const lower = modelName.toLowerCase();
  if (lower.includes('gpt') || lower.includes('openai')) return 'OpenAI';
  if (lower.includes('claude') || lower.includes('anthropic')) return 'Anthropic';
  if (lower.includes('gemini') || lower.includes('google')) return 'Google';
  if (lower.includes('grok') || lower.includes('xai')) return 'xAI';
  if (lower.includes('kimi') || lower.includes('moonshot')) return 'MoonshotAI';
  if (lower.includes('deepseek')) return 'DeepSeek';
  if (lower.includes('mistral')) return 'Mistral';
  if (lower.includes('minimax')) return 'MiniMax';
  if (lower.includes('qwen')) return 'Qwen';
  if (lower.includes('glm') || lower.includes('z.ai')) return 'Z.ai';
  return 'Other';
}

export function getShortName(modelName = '') {
  const match = modelName.match(/^[^:]+:\s*(.+)/);
  return match ? match[1].trim() : modelName;
}

export function filterModelsForRole(models, role, zdrEnabled) {
  const validTypes = role === 'chairman' ? ['chairman', 'both'] : ['council', 'both'];
  return (models || []).filter((model) => (
    validTypes.includes(model.type) && (!zdrEnabled || model.supports_zdr)
  ));
}

export function groupModelsByProvider(models) {
  const groups = new Map();
  for (const model of models || []) {
    const provider = getProvider(model.name);
    if (!groups.has(provider)) groups.set(provider, []);
    groups.get(provider).push(model);
  }
  return [...groups.entries()].sort((a, b) => b[1].length - a[1].length);
}
