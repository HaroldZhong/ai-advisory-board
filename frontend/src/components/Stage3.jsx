import { memo } from 'react';
import MarkdownRenderer from './MarkdownRenderer';
import ReasoningSection from './ReasoningSection';
import { Card } from "@/components/ui/card";
import { formatReasoningActuals } from "@/utils/reasoningDisplay";

function Stage3({
  finalResponse,
  evidenceSnapshot,
  onCitationClick,
  messageKey = 'message',
  showReasoningByDefault = false,
}) {
  if (!finalResponse) {
    return null;
  }

  const chairmanLabel = finalResponse.model.split('/')[1] || finalResponse.model;

  return (
    <div className="space-y-4">
      <h3 className="text-base font-semibold">Council answer</h3>

      <Card className="border border-primary/20 bg-background p-4 shadow-sm sm:p-5">
        <div className="mb-2 break-all text-xs font-semibold uppercase tracking-wide text-primary">
          Chairman: {chairmanLabel}
        </div>
        <ReasoningSection
          className="mb-4"
          modelId={finalResponse.model}
          modelLabel="Chairman"
          reasoningText={finalResponse.reasoning}
          status="complete"
          defaultExpanded={showReasoningByDefault}
          storageKey={`aab.reasoning.${messageKey}.stage3`}
        />
        <div className="prose max-w-none text-sm dark:prose-invert">
          <MarkdownRenderer evidenceSnapshot={evidenceSnapshot} onCitationClick={onCitationClick}>{finalResponse.response}</MarkdownRenderer>
        </div>
        {/* B5/E3 §3d: honest post-turn reasoning actuals, keyed on token count. */}
        <details className="mt-3 text-xs text-muted-foreground">
          <summary className="cursor-pointer">Model details</summary>
          <p>{formatReasoningActuals(finalResponse.reasoning_tokens)}</p>
        </details>
      </Card>
    </div>
  );
}

export default memo(Stage3);
