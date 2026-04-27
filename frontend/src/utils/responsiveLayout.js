export const RESPONSIVE_WIDTHS = {
  minimumDesktop: 960,
  fullDesktop: 1024,
};

export const SIDEBAR_COLLAPSED_STORAGE_KEY = 'aab.sidebar.collapsed';

export function parseSidebarCollapsedPreference(value) {
  return value === 'true';
}

export function readSidebarCollapsedPreference(storage) {
  try {
    const targetStorage = storage ?? globalThis.localStorage;
    return parseSidebarCollapsedPreference(targetStorage?.getItem(SIDEBAR_COLLAPSED_STORAGE_KEY));
  } catch {
    return false;
  }
}

export function writeSidebarCollapsedPreference(collapsed, storage) {
  try {
    const targetStorage = storage ?? globalThis.localStorage;
    targetStorage?.setItem(SIDEBAR_COLLAPSED_STORAGE_KEY, String(Boolean(collapsed)));
  } catch {
    // Storage can be unavailable in privacy-restricted or sandboxed contexts.
  }
}

export function getSidebarMode(width, userCollapsed = false) {
  const viewportWidth = Number(width || 0);
  if (viewportWidth < RESPONSIVE_WIDTHS.fullDesktop) return 'icon-only';
  if (userCollapsed) return 'collapsed';
  return 'expanded';
}
