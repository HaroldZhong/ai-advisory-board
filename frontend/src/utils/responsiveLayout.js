export const RESPONSIVE_WIDTHS = {
  minimumDesktop: 960,
  fullDesktop: 1024,
};

export const SIDEBAR_COLLAPSED_STORAGE_KEY = 'aab.sidebar.collapsed';

export function parseSidebarCollapsedPreference(value) {
  return value === 'true';
}

export function getSidebarMode(width, userCollapsed = false) {
  const viewportWidth = Number(width || 0);
  if (viewportWidth < RESPONSIVE_WIDTHS.fullDesktop) return 'icon-only';
  if (userCollapsed) return 'collapsed';
  return 'expanded';
}
