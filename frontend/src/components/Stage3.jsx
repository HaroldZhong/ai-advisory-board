import { memo } from 'react';
import MarkdownRenderer from './MarkdownRenderer';
import ReasoningSection from './ReasoningSection';
import { Card } from "@/components/ui/card";

function Stage3({
  finalResponse,
  messageKey = 'message',
  showReasoningByDefault = false,
}) {
  if (!finalResponse) {
    return null;
  }

  const chairmanLabel = finalResponse.model.split('/')[1] || finalResponse.model;

  return (
    <div className="space-y-4">
      <h3 className="flex flex-wrap items-center gap-2 text-lg font-semibold">
        <span className="rounded bg-primary/10 px-2 py-1 text-sm text-primary">Stage 3</span>
        <span>Final Council Answer</span>
      </h3>

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
          <MarkdownRenderer>{finalResponse.response}</MarkdownRenderer>
        </div>
      </Card>
    </div>
  );
}

export default memo(Stage3);
