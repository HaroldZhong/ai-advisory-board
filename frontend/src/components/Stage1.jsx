import { memo, useState } from 'react';
import MarkdownRenderer from './MarkdownRenderer';
import ReasoningSection from './ReasoningSection';
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { getStageTabListClass } from "@/utils/responsiveChatLayout";
import { formatReasoningActuals } from "@/utils/reasoningDisplay";

function Stage1({ responses, messageKey = 'message', showReasoningByDefault = false }) {
  const [activeTab, setActiveTab] = useState(0);

  if (!responses || responses.length === 0) {
    return null;
  }

  const activeResponse = responses[activeTab];
  const activeModelLabel = activeResponse.model.split('/')[1] || activeResponse.model;

  return (
    <div className="space-y-4">
      <h3 className="flex flex-wrap items-center gap-2 text-lg font-semibold">
        <span className="rounded bg-primary/10 px-2 py-1 text-sm text-primary">Stage 1</span>
        <span>Individual Responses</span>
      </h3>

      <div className={getStageTabListClass()}>
        {responses.map((resp, index) => (
          <Button
            key={index}
            variant={activeTab === index ? "default" : "outline"}
            size="sm"
            onClick={() => setActiveTab(index)}
            className="max-w-[12rem] shrink-0 truncate text-xs"
          >
            {resp.model.split('/')[1] || resp.model}
          </Button>
        ))}
      </div>

      <Card className="p-4 bg-background border">
        <div className="mb-2 break-all text-xs font-semibold text-muted-foreground">
          {activeResponse.model}
        </div>
        <ReasoningSection
          className="mb-4"
          modelId={activeResponse.model}
          modelLabel={activeModelLabel}
          reasoningText={activeResponse.reasoning}
          status="complete"
          defaultExpanded={showReasoningByDefault}
          storageKey={`aab.reasoning.${messageKey}.stage1.${activeTab}`}
        />
        <div className="prose max-w-none text-sm dark:prose-invert">
          <MarkdownRenderer>{activeResponse.response}</MarkdownRenderer>
        </div>
        {/* B5/E3 §3d: honest post-turn reasoning actuals, keyed on token count. */}
        <div className="mt-3 text-xs text-muted-foreground">
          {formatReasoningActuals(activeResponse.reasoning_tokens)}
        </div>
      </Card>
    </div>
  );
}

export default memo(Stage1);
