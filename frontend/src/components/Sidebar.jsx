import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { PlusCircle, BarChart2, MessageSquare } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Reusable sidebar content component - used in both desktop sidebar and mobile drawer
 */
export function SidebarContent({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  onShowAnalytics,
  onItemClick, // Optional callback for closing mobile drawer after selection
}) {
  const handleConversationSelect = (id) => {
    onSelectConversation(id);
    onItemClick?.();
  };

  const handleNewConversation = () => {
    onNewConversation();
    onItemClick?.();
  };

  const handleShowAnalytics = () => {
    onShowAnalytics();
    onItemClick?.();
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 space-y-4">
        <div className="flex items-center justify-between px-2">
          <h1 className="text-xl font-bold tracking-tight">AI Advisory Board</h1>
        </div>
        <Button onClick={handleNewConversation} className="w-full justify-start" size="lg">
          <PlusCircle className="mr-2 h-4 w-4" />
          New Conversation
        </Button>
        <Button onClick={handleShowAnalytics} variant="outline" className="w-full justify-start">
          <BarChart2 className="mr-2 h-4 w-4" />
          Analytics
        </Button>
      </div>

      <Separator />

      <ScrollArea className="flex-1 px-2 py-2">
        <div className="space-y-1 p-2">
          {conversations.length === 0 ? (
            <div className="p-4 text-sm text-muted-foreground text-center">
              No conversations yet
            </div>
          ) : (
            conversations.map((conv) => (
              <Button
                key={conv.id}
                variant={conv.id === currentConversationId ? "secondary" : "ghost"}
                className={cn(
                  "w-full justify-start font-normal h-auto py-3",
                  conv.id === currentConversationId && "bg-secondary"
                )}
                onClick={() => handleConversationSelect(conv.id)}
              >
                <div className="flex flex-col items-start gap-1 w-full overflow-hidden">
                  <div className="flex items-center w-full">
                    <MessageSquare className="mr-2 h-4 w-4 shrink-0 opacity-70" />
                    <span className="truncate">{conv.title || 'New Conversation'}</span>
                  </div>
                  <span className="text-xs text-muted-foreground ml-6">
                    {conv.message_count} messages
                  </span>
                </div>
              </Button>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
}

/**
 * Desktop sidebar wrapper - hidden on mobile
 */
export default function Sidebar(props) {
  return (
    <aside className="w-[300px] border-r bg-muted/10 hidden md:flex flex-col h-full">
      <SidebarContent {...props} />
    </aside>
  );
}
