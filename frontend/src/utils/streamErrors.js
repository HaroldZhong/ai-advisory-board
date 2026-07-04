/**
 * Turn a raw backend stream-error message into something a user can act on.
 * Backend sends {type:'error', message:<str(exception)>} (backend/main.py:1392).
 */
const NETWORK_HINTS = /connect|timeout|timed out|dns|unreachable|nodename|network/i;

export function formatStreamErrorMessage(rawMessage) {
  const raw = (rawMessage || '').trim();
  if (!raw) {
    return 'Something went wrong while generating the response. Please try again.';
  }
  if (NETWORK_HINTS.test(raw)) {
    return `Could not reach OpenRouter (network problem). Check your connection or proxy settings. Details: ${raw}`.slice(0, 300);
  }
  return raw.slice(0, 300);
}
