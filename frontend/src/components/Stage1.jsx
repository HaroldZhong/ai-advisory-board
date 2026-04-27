import { useState } from 'react';
import MarkdownRenderer from './MarkdownRenderer';
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { getStageTabListClass } from "@/utils/responsiveChatLayout";

export default function Stage1({ responses }) {
  const [activeTab, setActiveTab] = useState(0);

  if (!responses || responses.length === 0) {
    return null;
  }

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
          {responses[activeTab].model}
        </div>
        <div className="prose max-w-none text-sm dark:prose-invert">
          <MarkdownRenderer>{responses[activeTab].response}</MarkdownRenderer>
        </div>
      </Card>
    </div>
  );
}
