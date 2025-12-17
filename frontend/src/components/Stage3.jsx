import MarkdownRenderer from './MarkdownRenderer';
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export default function Stage3({ finalResponse }) {
  if (!finalResponse) {
    return null;
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold flex items-center gap-2">
        <span className="bg-primary/10 text-primary px-2 py-1 rounded text-sm">Stage 3</span>
        Final Council Answer
      </h3>

      <Card className="p-5 bg-background border border-primary/20 shadow-sm">
        <div className="text-xs font-semibold text-primary mb-2 uppercase tracking-wide">
          Chairman: {finalResponse.model.split('/')[1] || finalResponse.model}
        </div>
        <div className="prose max-w-none text-sm dark:prose-invert">
          <MarkdownRenderer>{finalResponse.response}</MarkdownRenderer>
        </div>
      </Card>
    </div>
  );
}
