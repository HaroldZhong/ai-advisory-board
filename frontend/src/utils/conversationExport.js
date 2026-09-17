export function getExportSavedDescription(path) {
  return path ? `Saved ${path.split(/[\\/]/).pop()}.` : 'Export saved.';
}
