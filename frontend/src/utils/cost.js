/**
 * Calculate cost in USD based on usage and model pricing.
 * @param {Object} usage - { prompt_tokens, completion_tokens }
 * @param {string} modelId - The model ID (e.g. "openai/gpt-4o")
 * @param {Array} modelsList - List of available models with pricing info
 * @returns {number} Cost in USD
 */
export function calculateUsageCost(usage, modelId, modelsList) {
    if (!usage || !modelId || !modelsList) return 0;

    const model = modelsList.find(m => m.id === modelId);
    if (!model || !model.pricing) return 0;

    const promptCost = (usage.prompt_tokens || 0) * (model.pricing.input / 1000000);
    const completionCost = (usage.completion_tokens || 0) * (model.pricing.output / 1000000);

    return promptCost + completionCost;
}

/**
 * Calculate total cost for a Stage 1 (array of responses).
 * @param {Array} results - Array of objects with { model, usage }
 * @param {Array} modelsList 
 * @returns {number}
 */
export function calculateStage1Cost(results, modelsList) {
    if (!results) return 0;
    return results.reduce((acc, res) => {
        return acc + calculateUsageCost(res.usage, res.model, modelsList);
    }, 0);
}

/**
 * Calculate total cost for Stage 2 (array of rankings).
 * @param {Array} results - Array of objects with { model, usage }
 * @param {Array} modelsList 
 * @returns {number}
 */
export function calculateStage2Cost(results, modelsList) {
    // Same structure as stage 1
    return calculateStage1Cost(results, modelsList);
}

/**
 * Calculate cost for Stage 3 (single result).
 * @param {Object} result - { model, usage }
 * @param {Array} modelsList 
 * @returns {number}
 */
export function calculateStage3Cost(result, modelsList) {
    if (!result) return 0;
    return calculateUsageCost(result.usage, result.model, modelsList);
}
