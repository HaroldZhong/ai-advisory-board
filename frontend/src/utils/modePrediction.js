/**
 * Predict which mode the backend will resolve for the next "auto" send.
 *
 * ROUTING IS BACKEND-OWNED (P3-T1): the wire request always carries
 * mode:"auto" and backend/main.py prepare_turn makes the authoritative,
 * edit-aware decision. This prediction exists ONLY for presentation —
 * choosing the optimistic assistant skeleton and which advanced settings
 * apply — and mirrors the backend rule exactly.
 */
export function predictNextMessageMode({ messageCount = 0, editIndex = -1 } = {}) {
  const effectiveCount = editIndex >= 0 ? editIndex : messageCount;
  return effectiveCount === 0 ? 'council' : 'chat';
}
