export function isLandingOnly(env = import.meta.env) {
  return String(env.VITE_LANDING_ONLY).toLowerCase() === 'true';
}

export function resolveAppEntryTarget(env = import.meta.env) {
  // Landing-only builds send "launch" CTAs to the release download instead of /app.
  return isLandingOnly(env)
    ? 'https://github.com/HaroldZhong/ai-advisory-board/releases'
    : '/app';
}
