import { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import { languageFromClassName } from '../utils/codeLanguage';

export { languageFromClassName };

/**
 * <pre> override: fenced code blocks get a language label + copy button.
 * Inline code (no language className on the child <code>) falls back to
 * a plain <pre> so inline styling from index.css still applies.
 */
export function CodeBlock({ children, ...rest }) {
    const codeElement = Array.isArray(children) ? children[0] : children;
    const className = codeElement?.props?.className;
    const language = languageFromClassName(className);
    const rawText = codeElement?.props?.children;
    const codeText = Array.isArray(rawText) ? rawText.join('') : String(rawText ?? '');

    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(codeText);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
        } catch (err) {
            console.warn('Copy to clipboard failed:', err);
        }
    };

    return (
        <div className="group relative my-4">
            <div className="flex items-center justify-between rounded-t-md border border-b-0 border-border bg-[#161b22] px-3 py-1.5 text-xs text-gray-300">
                <span className="font-mono lowercase">{language || 'text'}</span>
                <button
                    type="button"
                    onClick={handleCopy}
                    className="flex items-center gap-1 rounded px-1.5 py-0.5 text-gray-300 transition-colors hover:bg-white/10 hover:text-white"
                >
                    {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
                    {copied ? 'Copied!' : 'Copy'}
                </button>
            </div>
            <pre {...rest} className="!my-0 !rounded-t-none">
                {children}
            </pre>
        </div>
    );
}
