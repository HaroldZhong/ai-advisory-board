import { memo, useState } from 'react';
import MarkdownRenderer from './MarkdownRenderer';
import ReasoningSection from './ReasoningSection';
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { getStageTabListClass } from "@/utils/responsiveChatLayout";

function deAnonymizeText(text, labelToModel) {
  if (!labelToModel) return text;

  let result = text;
  Object.entries(labelToModel).forEach(([label, model]) => {
    const modelShortName = model.split('/')[1] || model;
    result = result.replace(new RegExp(label, 'g'), `**${modelShortName}**`);
  });
  return result;
}

function Stage2({
  rankings,
  labelToModel,
  aggregateRankings,
  messageKey = 'message',
  showReasoningByDefault = false,
}) {
  const [activeTab, setActiveTab] = useState(0);

  if (!rankings || rankings.length === 0) {
    return null;
  }

  const activeRanking = rankings[activeTab];
  const activeModelLabel = activeRanking.model.split('/')[1] || activeRanking.model;

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h3 className="flex flex-wrap items-center gap-2 text-lg font-semibold">
          <span className="rounded bg-primary/10 px-2 py-1 text-sm text-primary">Stage 2</span>
          <span>Peer Rankings</span>
        </h3>
        <p className="text-sm text-muted-foreground">
          Council models evaluate the responses below. If an evaluator is unavailable, it remains visible here.
        </p>
      </div>

      <div className="space-y-4">
        <h4 className="text-sm font-semibold">Raw Evaluations</h4>
        <div className={getStageTabListClass()}>
          {rankings.map((rank, index) => (
            <Button
              key={index}
              variant={activeTab === index ? "default" : "outline"}
              size="sm"
              onClick={() => setActiveTab(index)}
              className="max-w-[12rem] shrink-0 truncate text-xs"
            >
              {rank.model.split('/')[1] || rank.model}
            </Button>
          ))}
        </div>

        <Card className="p-4 bg-background border">
          <div className="mb-2 break-all text-xs font-semibold text-muted-foreground">
            Evaluator: {activeRanking.model}
          </div>
          <ReasoningSection
            className="mb-4"
            modelId={activeRanking.model}
            modelLabel={activeModelLabel}
            reasoningText={activeRanking.reasoning}
            status="complete"
            defaultExpanded={showReasoningByDefault}
            storageKey={`aab.reasoning.${messageKey}.stage2.${activeTab}`}
          />
          <div className="prose max-w-none text-sm dark:prose-invert mb-4">
            <MarkdownRenderer>
              {deAnonymizeText(activeRanking.ranking, labelToModel)}
            </MarkdownRenderer>
          </div>

          {activeRanking.parsed_ranking && activeRanking.parsed_ranking.length > 0 && (
            <div className="bg-muted/50 p-3 rounded-md">
              <strong className="text-xs uppercase tracking-wider text-muted-foreground">Extracted Ranking</strong>
              <ol className="list-decimal list-inside text-sm mt-1 space-y-1">
                {activeRanking.parsed_ranking.map((label, i) => (
                  <li key={i}>
                    {labelToModel && labelToModel[label]
                      ? labelToModel[label].split('/')[1] || labelToModel[label]
                      : label}
                  </li>
                ))}
              </ol>
            </div>
          )}
        </Card>
      </div>

      {aggregateRankings && aggregateRankings.length > 0 && (
        <div className="space-y-4">
          <h4 className="text-sm font-semibold">Aggregate Rankings (Street Cred)</h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {aggregateRankings.map((agg, index) => (
              <Card key={index} className="p-3 flex items-center gap-3">
                <div className="flex items-center justify-center h-8 w-8 rounded-full bg-primary/10 text-primary font-bold text-sm">
                  #{index + 1}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="font-medium truncate text-sm">
                    {agg.model.split('/')[1] || agg.model}
                  </div>
                  <div className="flex flex-wrap gap-x-2 text-xs text-muted-foreground">
                    <span>Avg: {agg.average_rank.toFixed(2)}</span>
                    <span>({agg.rankings_count} votes)</span>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default memo(Stage2);
