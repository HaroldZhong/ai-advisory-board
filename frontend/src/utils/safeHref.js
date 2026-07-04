/**
 * Allowlist URL schemes for model-generated links (audit §7).
 * Browsers strip ASCII control chars and spaces when resolving URLs
 * (so `java\nscript:` executes) — strip them BEFORE scheme detection.
 */
const ALLOWED_SCHEMES = new Set(['http', 'https', 'mailto']);
const SCHEME = /^([a-zA-Z][a-zA-Z0-9+.-]*):/;

export function isSafeHref(href) {
  if (typeof href !== 'string' || href.length === 0) return false;
  const cleaned = href.replace(/[\u0000-\u0020]/g, '');
  if (cleaned.length === 0) return false;
  const match = SCHEME.exec(cleaned);
  if (!match) return true; // relative URL, #anchor, or //protocol-relative
  return ALLOWED_SCHEMES.has(match[1].toLowerCase());
}
