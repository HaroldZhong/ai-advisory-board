import { useState } from 'react';
import MarkdownRenderer from './MarkdownRenderer';
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

function deAnonymizeText(text, labelToModel) {
  if (!labelToModel) return text;

  let result = text;
  Object.entries(labelToModel).forEach(([label, model]) => {
    const modelShortName = model.split('/')[1] || model;
    result = result.replace(new RegExp(label, 'g'), `**${modelShortName}**`);
  });
  return result;
}

export default function Stage2({ rankings, labelToModel, aggregateRankings }) {
  const [activeTab, setActiveTab] = useState(0);

  if (!rankings || rankings.length === 0) {
    return null;
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <span className="bg-primary/10 text-primary px-2 py-1 rounded text-sm">Stage 2</span>
          Peer Rankings
        </h3>
        <p className="text-sm text-muted-foreground">
          Each model evaluated all responses. Below are the individual evaluations and the extracted rankings.
        </p>
      </div>

      <div className="space-y-4">
        <h4 className="text-sm font-semibold">Raw Evaluations</h4>
        <div className="flex flex-wrap gap-2">
          {rankings.map((rank, index) => (
            <Button
              key={index}
              variant={activeTab === index ? "default" : "outline"}
              size="sm"
              onClick={() => setActiveTab(index)}
              className="text-xs"
            >
              {rank.model.split('/')[1] || rank.model}
            </Button>
          ))}
        </div>

        <Card className="p-4 bg-background border">
          <div className="text-xs font-semibold text-muted-foreground mb-2">
            Evaluator: {rankings[activeTab].model}
          </div>
          <div className="prose max-w-none text-sm dark:prose-invert mb-4">
            <MarkdownRenderer>
              {deAnonymizeText(rankings[activeTab].ranking, labelToModel)}
            </MarkdownRenderer>
          </div>

          {rankings[activeTab].parsed_ranking && rankings[activeTab].parsed_ranking.length > 0 && (
            <div className="bg-muted/50 p-3 rounded-md">
              <strong className="text-xs uppercase tracking-wider text-muted-foreground">Extracted Ranking</strong>
              <ol className="list-decimal list-inside text-sm mt-1 space-y-1">
                {rankings[activeTab].parsed_ranking.map((label, i) => (
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
                  <div className="text-xs text-muted-foreground flex gap-2">
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
