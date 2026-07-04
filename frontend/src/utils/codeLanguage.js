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

/**
 * Flatten a React children tree back to plain text. rehype-highlight
 * wraps tokens in <span> elements, so a highlighted code block's
 * children are nested element-like objects, not plain strings -- this
 * walks them the same way React would render them. Pure + node-testable
 * (element-like objects here just need a `.props.children` field, not
 * real React elements).
 */
export function extractNodeText(node) {
    if (node == null || typeof node === 'boolean') return '';
    if (typeof node === 'string' || typeof node === 'number') return String(node);
    if (Array.isArray(node)) return node.map(extractNodeText).join('');
    if (typeof node === 'object' && node.props) return extractNodeText(node.props.children);
    return '';
}
