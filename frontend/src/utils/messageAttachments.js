/**
 * Extract the attachment ids a persisted message carries, so Edit &
 * Regenerate can resend them (P3-T8 item 3 — the resend used to hardcode an
 * empty attachment list and silently drop them).
 *
 * Storage persists both `attachment_ids` (raw ids) and `attachments` (rich
 * metadata) on a user message (backend/storage.py add_user_message) —
 * prefer the raw id list, falling back to deriving ids from the metadata.
 */
export function extractMessageAttachmentIds(message) {
  if (!message) return [];
  if (Array.isArray(message.attachment_ids)) return message.attachment_ids;
  if (Array.isArray(message.attachments)) {
    return message.attachments
      .map((attachment) => attachment?.attachment_id)
      .filter(Boolean);
  }
  return [];
}
