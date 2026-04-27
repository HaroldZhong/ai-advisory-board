export const RESPONSIVE_WIDTHS = {
  minimumDesktop: 960,
  fullDesktop: 1024,
};

export function getSidebarMode(width, userCollapsed = false) {
  const viewportWidth = Number(width || 0);
  if (viewportWidth < RESPONSIVE_WIDTHS.fullDesktop) return 'icon-only';
  if (userCollapsed) return 'collapsed';
  return 'expanded';
}
