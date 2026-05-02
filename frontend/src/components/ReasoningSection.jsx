import { useEffect, useId, useState } from 'react';
import { Brain, ChevronDown, Loader2 } from 'lucide-react';
import MarkdownRenderer from './MarkdownRenderer';
import { cn } from '@/lib/utils';
import {
  formatDuration,
  formatTokenCount,
  getReasoningStatusLabel,
  hasReasoningText,
} from '@/utils/reasoningDisplay';

function readExpandedPreference(storageKey, fallback) {
  if (!storageKey) return fallback;
  if (typeof window === 'undefined') return fallback;

  try {
    const stored = window.localStorage?.getItem(storageKey);
    if (stored === 'true') return true;
    if (stored === 'false') return false;
  } catch {
    // Non-critical UI preference; blocked storage should not hide reasoning.
  }

  return fallback;
}

function writeExpandedPreference(storageKey, expanded) {
  if (!storageKey) return;
  if (typeof window === 'undefined') return;

  try {
    window.localStorage?.setItem(storageKey, expanded ? 'true' : 'false');
  } catch {
    // Non-critical UI preference; ignore unavailable or blocked storage.
  }
}

function getReasoningExcerpt(reasoningText) {
  const compact = reasoningText.replace(/\s+/g, ' ').trim();
  if (compact.length <= 140) return compact;
  return `${compact.slice(0, 137)}...`;
}

export default function ReasoningSection({
  modelId,
  modelLabel = 'Model',
  reasoningText,
  status = 'complete',
  tokenCount,
  durationMs,
  defaultExpanded = false,
  storageKey,
  className,
}) {
  const reactId = useId();
  const bodyId = `reasoning-${reactId.replace(/:/g, '')}`;
  const [isExpanded, setIsExpanded] = useState(() => (
    readExpandedPreference(storageKey, defaultExpanded)
  ));

  useEffect(() => {
    setIsExpanded(readExpandedPreference(storageKey, defaultExpanded));
  }, [defaultExpanded, storageKey]);

  if (!hasReasoningText(reasoningText)) {
    return null;
  }

  const statusLabel = getReasoningStatusLabel(status);
  const durationLabel = formatDuration(durationMs);
  const tokenLabel = formatTokenCount(tokenCount);
  const metaItems = [durationLabel, tokenLabel].filter(Boolean);
  const isStreaming = status === 'streaming';

  const toggleExpanded = () => {
    setIsExpanded((current) => {
      const next = !current;
      writeExpandedPreference(storageKey, next);
      return next;
    });
  };

  return (
    <section className={cn("rounded-lg border border-violet-500/20 bg-violet-500/[0.03]", className)}>
      <button
        type="button"
        onClick={toggleExpanded}
        aria-expanded={isExpanded}
        aria-controls={bodyId}
        className={cn(
          "flex w-full items-start gap-3 rounded-lg p-3 text-left transition-colors",
          "hover:bg-violet-500/[0.06]",
          isExpanded && "rounded-b-none"
        )}
      >
        <div className="relative mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-violet-500/10 text-violet-600 dark:text-violet-300">
          {isStreaming ? (
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          ) : (
            <Brain className="h-4 w-4" aria-hidden="true" />
          )}
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-foreground">{statusLabel}</span>
            <span className="rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">
              {modelLabel}
            </span>
            {metaItems.length > 0 && (
              <span className="text-xs text-muted-foreground">
                {metaItems.join(' · ')}
              </span>
            )}
          </div>
          {modelId && (
            <div className="mt-0.5 break-all font-mono text-[11px] text-muted-foreground">
              {modelId}
            </div>
          )}
          {!isExpanded && (
            <p className="mt-1 truncate text-xs text-muted-foreground">
              {getReasoningExcerpt(reasoningText)}
            </p>
          )}
        </div>

        <ChevronDown
          className={cn(
            "mt-1 h-4 w-4 shrink-0 text-muted-foreground transition-transform",
            isExpanded && "rotate-180"
          )}
          aria-hidden="true"
        />
      </button>

      {isExpanded && (
        <div
          id={bodyId}
          className="max-h-[24rem] overflow-y-auto border-t border-violet-500/20 p-3"
        >
          <div className="prose prose-sm max-w-none dark:prose-invert">
            <MarkdownRenderer>{reasoningText}</MarkdownRenderer>
          </div>
        </div>
      )}
    </section>
  );
}
