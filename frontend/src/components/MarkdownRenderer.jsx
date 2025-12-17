import React, { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
// Note: katex.min.css is loaded globally in main.jsx

/**
 * MarkdownRenderer - Enhanced markdown with LaTeX equation support
 * 
 * Handles multiple equation formats from different LLM providers:
 * - Standard LaTeX: $inline$ and $$block$$
 * - Bracket notation: \( inline \) and \[ block \]
 * 
 * Uses conservative preprocessing to avoid false positives.
 */

// KaTeX options - graceful error handling
const katexOptions = {
    throwOnError: false,  // Don't crash on parse errors
    errorColor: '#666',   // Gray color for error text
    strict: false,        // Lenient parsing
    trust: true,          // Allow all LaTeX commands
    output: 'htmlAndMathml', // Better accessibility
};

/**
 * Pre-process content to normalize equation delimiters
 * Only handles standard LaTeX notation to avoid false positives
 */
const preprocessMath = (content) => {
    if (!content || typeof content !== 'string') return content || '';

    // Robust tokenizer pattern
    // Order matters! Captured groups are protected.
    // 1. Code blocks (```...```)
    // 2. Inline code (`...`)
    // 3. Images (![...](...))
    // 4. Links ([...](...)) - checking ](
    // 5. Display math ($$...$$)
    // 6. Display math (\[...\])
    // 7. Inline math ($...$)
    // 8. Inline math (\(...\))
    const tokenRegex = /((?:^|\n)```[\s\S]*?(?:$|\n```)|`[^`]*`|!\[[^\]]*\]\([^)]+\)|\[[^\]]*\]\([^)]+\)|\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\$(?!$)[^$]+?\$|\\\([\s\S]*?\\\))/g;

    const parts = content.split(tokenRegex);

    return parts.map(part => {
        // If part looks like a protected token, handle appropriately
        if (!part) return '';
        
        // Code blocks, inline code, images, links - return unchanged
        if (
            part.startsWith('```') ||
            part.startsWith('`') ||
            part.startsWith('![') ||
            (part.startsWith('[') && part.includes(']('))
        ) {
            return part;
        }
        
        // Already in dollar notation - return unchanged
        if (part.startsWith('$$') || part.startsWith('$')) {
            return part;
        }
        
        // Convert \[...\] to $$...$$ (display math)
        if (part.startsWith('\\[') && part.endsWith('\\]')) {
            return '$$' + part.slice(2, -2) + '$$';
        }
        
        // Convert \(...\) to $...$ (inline math)
        if (part.startsWith('\\(') && part.endsWith('\\)')) {
            return '$' + part.slice(2, -2) + '$';
        }

        // Process Text Chunk - MINIMAL heuristics only
        // Most math should come through with proper \(...\) or \[...\] delimiters
        // which are converted above. Only apply safe transformations here.
        let processed = part;

        // 1. Formatting cleanup for markdown headings
        processed = processed.replace(/(\n)\s*(?=#{1,6}\s|---)/g, '\n\n');
        processed = processed.replace(/([^\n])\s*(?=#{1,6}\s|---)/g, '$1\n\n');

        // 2. Only convert standalone LaTeX commands that are unambiguously math
        // e.g., \alpha, \beta, \pi (Greek letters commonly used in math)
        // Must be preceded by whitespace or start of string, followed by word boundary
        processed = processed.replace(/(^|[\s(])\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|infty|sum|prod|int|partial|nabla|pm|mp|times|cdot|leq|geq|neq|approx|equiv|subset|supset|in|notin|forall|exists|rightarrow|leftarrow|Rightarrow|Leftarrow|leftrightarrow)(\b|[^a-zA-Z])/g, 
            (match, pre, cmd, post) => `${pre}$\\${cmd}$${post}`
        );

        return processed;
    }).join('');
};

/**
 * Error boundary for catching render errors
 */
class MarkdownErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false };
    }

    static getDerivedStateFromError() {
        return { hasError: true };
    }

    componentDidCatch(error, errorInfo) {
        console.warn('MarkdownRenderer error:', error.message);
    }

    render() {
        if (this.state.hasError) {
            // Unintrusive error display
            return (
                <div className={`text-red-500 text-xs p-2 border border-red-200 rounded bg-red-50/50 dark:bg-red-900/10 dark:border-red-900/50 ${this.props.className}`}>
                    <div className="font-semibold mb-1">Rendering Error</div>
                    <pre className="whitespace-pre-wrap font-mono text-[10px] opacity-75">
                        {this.props.rawContent.substring(0, 100)}...
                    </pre>
                </div>
            );
        }
        return this.props.children;
    }
}

/**
 * MarkdownRenderer component with LaTeX support
 * 
 * Features:
 * - Memoized content processing for performance
 * - Graceful fallback on errors
 * - Handles multiple LaTeX formats
 */
export default function MarkdownRenderer({
    children,
    className = '',
    components = {},
    ...props
}) {
    // Handle null/undefined content gracefully
    if (children == null || children === '') {
        return null;
    }

    // Ensure we have a string
    const rawContent = typeof children === 'string' ? children : String(children);

    // Memoize the processed content to prevent unnecessary re-renders
    const processedContent = useMemo(() => {
        return preprocessMath(rawContent);
    }, [rawContent]);

    return (
        <MarkdownErrorBoundary className={className} rawContent={rawContent}>
            {/* Add overflow protection at container level */}
            <div className={`w-full overflow-x-auto ${className}`}>
                <ReactMarkdown
                    remarkPlugins={[remarkMath]}
                    rehypePlugins={[[rehypeKatex, katexOptions]]}
                    components={components}
                    {...props}
                >
                    {processedContent}
                </ReactMarkdown>
            </div>
        </MarkdownErrorBoundary>
    );
}

/**
 * Lightweight version without math processing (for simple text)
 */
export function SimpleMarkdown({ children, className = '', ...props }) {
    if (children == null || children === '') {
        return null;
    }

    return (
        <div className={className}>
            <ReactMarkdown {...props}>
                {children}
            </ReactMarkdown>
        </div>
    );
}
