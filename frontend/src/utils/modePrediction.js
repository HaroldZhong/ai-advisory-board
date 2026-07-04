/**
 * Predict which mode the backend will resolve for the next "auto" send.
 *
 * ROUTING IS BACKEND-OWNED (P3-T1): the wire request always carries
 * mode:"auto" and backend/main.py prepare_turn makes the authoritative,
 * edit-aware decision. This prediction exists ONLY for presentation —
 * choosing the optimistic assistant skeleton and which advanced settings
 * apply — and mirrors the backend rule exactly, including default_mode
 * (P3-T3, master plan P3-W2, owner decision #2: Chat default, Council explicit).
 */
export function predictNextMessageMode({ messageCount = 0, editIndex = -1, defaultMode } = {}) {
  if (defaultMode === 'chat') return 'chat';

  // Clamped like the backend: truncation keeps messages[:editIndex], so a
  // stale editIndex beyond the stored count still leaves messageCount messages.
  const effectiveCount = editIndex >= 0 ? Math.min(editIndex, messageCount) : messageCount;
  return effectiveCount === 0 ? 'council' : 'chat';
}

/**
 * Resolve the optimistic skeleton/advanced-settings mode for a send, given an
 * optional explicit mode override (P3-T4: "Ask the council" on any turn).
 * An explicit mode (from the composer's council toggle) always wins over the
 * prediction — mirrors backend/main.py prepare_turn's "explicit request.mode
 * wins" rule.
 */
export function resolveSendMode(explicitMode, predictionArgs) {
  return explicitMode || predictNextMessageMode(predictionArgs);
}
