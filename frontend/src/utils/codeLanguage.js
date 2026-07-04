/**
 * Extract the language name from a rehype/remark code className
 * (e.g. "language-python" -> "python"). Pure + node-testable.
 */
export function languageFromClassName(className) {
    if (typeof className !== 'string' || className.length === 0) return null;
    const match = className.split(/\s+/).find((cls) => cls.startsWith('language-'));
    if (!match) return null;
    const lang = match.slice('language-'.length);
    return lang.length > 0 ? lang : null;
}
