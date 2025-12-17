import { useState } from 'react';
import MarkdownRenderer from './MarkdownRenderer';
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export default function Stage1({ responses }) {
  const [activeTab, setActiveTab] = useState(0);

  if (!responses || responses.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold flex items-center gap-2">
        <span className="bg-primary/10 text-primary px-2 py-1 rounded text-sm">Stage 1</span>
        Individual Responses
      </h3>

      <div className="flex flex-wrap gap-2">
        {responses.map((resp, index) => (
          <Button
            key={index}
            variant={activeTab === index ? "default" : "outline"}
            size="sm"
            onClick={() => setActiveTab(index)}
            className="text-xs"
          >
            {resp.model.split('/')[1] || resp.model}
          </Button>
        ))}
      </div>

      <Card className="p-4 bg-background border">
        <div className="text-xs font-semibold text-muted-foreground mb-2">
          {responses[activeTab].model}
        </div>
        <div className="prose max-w-none text-sm dark:prose-invert">
          <MarkdownRenderer>{responses[activeTab].response}</MarkdownRenderer>
        </div>
      </Card>
    </div>
  );
}
