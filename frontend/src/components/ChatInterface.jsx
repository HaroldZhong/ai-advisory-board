import { useState, useEffect, useRef } from 'react';
import MarkdownRenderer from './MarkdownRenderer';
import Stage1 from './Stage1';
import Stage2 from './Stage2';
import Stage3 from './Stage3';
import SessionBudgetSelector from './SessionBudgetSelector';
import AdvancedSettingsPanel from './AdvancedSettingsPanel';
import { api } from '../api';
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Paperclip, Send, Download, Loader2, Users, User, Crown, ChevronDown, Brain, Sparkles, Pencil } from "lucide-react";
import { cn } from "@/lib/utils";
import AttachmentPill, { AttachmentPillList } from './AttachmentPill';
import { useSettings } from '@/contexts/SettingsContext';
import TrustRow from './TrustRow';
import { getChatSurfaceClass } from '@/utils/responsiveChatLayout';
import { getBudgetCapBlockState, getPrivacyToggleDisabledReason, resolveEffectiveZdr } from '@/utils/trustState';

// Modern Chain of Thought component (ChatGPT/Claude style)
function ChainOfThought({ reasoning }) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!reasoning) return null;

  // Calculate summary stats
  const wordCount = reasoning.split(/\s+/).filter(Boolean).length;
  const lines = reasoning.split('\n').filter(Boolean).length;

  // Get a brief excerpt (first sentence or first 100 chars)
  const getExcerpt = () => {
    const firstSentence = reasoning.match(/^[^.!?]*[.!?]/);
    if (firstSentence && firstSentence[0].length < 150) {
      return firstSentence[0];
    }
    return reasoning.substring(0, 100) + '...';
  };

  return (
    <div className="relative">
      {/* Collapsible Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={cn(
          "w-full flex items-center gap-3 p-3 rounded-lg transition-all duration-200",
          "bg-gradient-to-r from-violet-500/10 via-purple-500/10 to-fuchsia-500/10",
          "hover:from-violet-500/15 hover:via-purple-500/15 hover:to-fuchsia-500/15",
          "border border-violet-500/20",
          isExpanded ? "rounded-b-none" : ""
        )}
      >
        {/* Thinking Icon */}
        <div className="relative shrink-0">
          <div className="h-8 w-8 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
            <Brain className="h-4 w-4 text-white" />
          </div>
          <Sparkles className="absolute -top-1 -right-1 h-3 w-3 text-violet-400" />
        </div>

        {/* Title and Summary */}
        <div className="flex-1 text-left min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-sm text-foreground">Reasoning</span>
            <span className="text-xs text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
              {wordCount} words
            </span>
          </div>
          {!isExpanded && (
            <p className="text-xs text-muted-foreground truncate mt-0.5">
              {getExcerpt()}
            </p>
          )}
        </div>

        {/* Expand/Collapse Icon */}
        <ChevronDown
          className={cn(
            "h-4 w-4 text-muted-foreground shrink-0 transition-transform duration-200",
            isExpanded && "rotate-180"
          )}
        />
      </button>

      {/* Expandable Content */}
      <div
        className={cn(
          "overflow-hidden transition-all duration-300 ease-in-out",
          isExpanded ? "max-h-[500px] opacity-100" : "max-h-0 opacity-0"
        )}
      >
        <div className={cn(
          "p-4 rounded-b-lg border border-t-0 border-violet-500/20",
          "bg-gradient-to-b from-violet-500/5 to-transparent"
        )}>
          <ScrollArea className="max-h-[400px]">
            <div className="prose prose-sm max-w-none dark:prose-invert text-muted-foreground">
              <MarkdownRenderer>{reasoning}</MarkdownRenderer>
            </div>
          </ScrollArea>
        </div>
      </div>
    </div>
  );
}

// Stage progress component with pulsing animation
function StageProgress({ stage, description, modelCount, icon: Icon }) {
  return (
    <div className="flex flex-col gap-3 rounded-lg border border-primary/20 bg-muted/50 p-3 sm:flex-row sm:items-center">
      <div className="relative shrink-0">
        <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
          <Icon className="h-5 w-5 text-primary" />
        </div>
        <div className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-primary flex items-center justify-center">
          <Loader2 className="h-3 w-3 animate-spin text-primary-foreground" />
        </div>
      </div>
      <div className="min-w-0 flex-1">
        <div className="font-medium text-sm">{stage}</div>
        <div className="text-xs text-muted-foreground">{description}</div>
      </div>
      {modelCount && (
        <div className="flex w-fit items-center gap-1.5 rounded border bg-background px-2 py-1 font-mono text-xs sm:shrink-0">
          <Users className="h-3 w-3" />
          <span className="text-muted-foreground">Processing</span>
          <span className="font-semibold text-primary">{modelCount}</span>
          <span className="text-muted-foreground">models</span>
        </div>
      )}
      <div className="flex gap-1 sm:shrink-0">
        {[...Array(3)].map((_, i) => (
          <div
            key={i}
            className="w-2 h-2 rounded-full bg-primary animate-pulse"
            style={{ animationDelay: `${i * 0.2}s` }}
          />
        ))}
      </div>
    </div>
  );
}

export default function ChatInterface({
  conversation,
  onSendMessage,
  onUpdateSessionPolicy,
  onUpdateConversationPrivacy,
  budgetWarning,
  isLoading,
}) {
  const [input, setInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isUpdatingPrivacy, setIsUpdatingPrivacy] = useState(false);
  const [attachments, setAttachments] = useState([]);  // Uploaded attachment metadata
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const [showBudgetSelector, setShowBudgetSelector] = useState(false);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [sendError, setSendError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [editingIndex, setEditingIndex] = useState(-1);
  const [editingContent, setEditingContent] = useState('');
  const { settings, updateSettings } = useSettings();

  const sessionPolicy = conversation?.session_policy || {};
  const sessionBudget = sessionPolicy.budget_usd ?? null;
  const nextMessageMode = conversation?.messages?.length > 0 ? 'chat' : 'council';
  const effectiveZdr = resolveEffectiveZdr(conversation, settings);
  const budgetCapBlock = getBudgetCapBlockState(conversation);
  const composerDisabled = isLoading || isUploading || isUpdatingPrivacy || budgetCapBlock.blocked;
  const privacyDisabledReason = getPrivacyToggleDisabledReason({
    isLoading,
    isUploading,
    isUpdatingPrivacy,
  });

  const handleBudgetConfirm = async (budgetUsd) => {
    try {
      setSendError(null);
      await onUpdateSessionPolicy?.({ budget_usd: budgetUsd });
    } catch (error) {
      alert(`Failed to update session budget: ${error.message || 'Unknown error'}`);
    }
  };

  // Robust scroll logic
  const viewportRef = useRef(null);
  const isNearBottomRef = useRef(true);
  const isRecentUpdate = useRef(Date.now());

  // Update timestamp when conversation changes
  useEffect(() => {
    isRecentUpdate.current = Date.now();
  }, [conversation?.messages?.length, conversation?.id]);

  // Track user scroll intent
  const handleScroll = (event) => {
    const viewport = event.target;
    if (!viewport) return;
    const { scrollTop, scrollHeight, clientHeight } = viewport;
    // User is "near bottom" if within 100px of the end
    isNearBottomRef.current = scrollHeight - scrollTop - clientHeight < 100;
  };

  const scrollToBottom = (behavior = 'smooth') => {
    const viewport = viewportRef.current;
    if (viewport) {
      viewport.scrollTo({ top: viewport.scrollHeight, behavior });
    }
  };

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;

    let observer = null;

    // 1. ResizeObserver to handle dynamic content (LaTeX, images)
    // Guard: Only use if ResizeObserver is available
    if (typeof ResizeObserver !== 'undefined') {
      let lastHeight = viewport.scrollHeight;

      observer = new ResizeObserver(() => {
        const currentHeight = viewport.scrollHeight;
        const heightIncreased = currentHeight > lastHeight + 10;

        // Only auto-scroll if:
        // A. We are properly streaming (isLoading)
        // B. OR we are in the initial "settling" phase (< 2s after load)
        const isSettling = Date.now() - isRecentUpdate.current < 2000;

        if (heightIncreased && isNearBottomRef.current && (isLoading || isSettling)) {
          scrollToBottom('instant');
        }
        lastHeight = currentHeight;
      });

      // Observe the content div (Radix puts content in a div inside viewport)
      const content = viewport.firstElementChild;
      if (content) observer.observe(content);
    }

    // 2. Handle new messages (standard React flow)
    if (isNearBottomRef.current) {
      scrollToBottom('smooth');
    }

    // 3. Safety check for cached font loading (KaTeX)
    // Guard: Only use if Font Loading API is available
    if (document.fonts?.ready) {
      document.fonts.ready.then(() => {
        if (isNearBottomRef.current) {
          viewport.scrollTo({ top: viewport.scrollHeight, behavior: 'auto' });
        }
      });
    }

    return () => {
      if (observer) observer.disconnect();
    };
  }, [conversation?.messages?.length, conversation?.id, isLoading]);

  const handleFileUpload = async (e) => {
    const selectedFiles = Array.from(e.target.files || []);
    if (selectedFiles.length === 0) return;
    if (isUpdatingPrivacy) {
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }

    const MAX_FILE_SIZE = 50 * 1024 * 1024;  // 50MB
    const MAX_FILES = 10;

    const oversized = selectedFiles.filter(f => f.size > MAX_FILE_SIZE);
    if (oversized.length > 0) {
      alert(`Some files exceed ${MAX_FILE_SIZE / 1024 / 1024}MB limit: ${oversized.map(f => f.name).join(', ')}`);
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }

    if (attachments.length + selectedFiles.length > MAX_FILES) {
      alert(`Maximum ${MAX_FILES} files allowed`);
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }

    // Upload files using new attachment API
    setIsUploading(true);
    try {
      for (const file of selectedFiles) {
        const result = await api.uploadAttachment(file, effectiveZdr);
        // Add to attachments with the metadata from the API
        setAttachments(prev => [...prev, {
          attachment_id: result.attachment_id,
          filename: result.filename,
          status: result.status,
          warning: result.warning,
          stats: result.stats,
          cached: result.cached,
          mime_type: file.type
        }]);
      }
    } catch (error) {
      alert(error.message);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const removeAttachment = async (indexToRemove) => {
    const attachment = attachments[indexToRemove];
    if (!attachment) return;

    try {
      if (attachment.attachment_id) {
        await api.deleteAttachment(attachment.attachment_id);
      }
      setAttachments(prev => prev.filter((_, index) => index !== indexToRemove));
    } catch (error) {
      console.error('Failed to delete attachment:', error);
      alert(`Failed to delete attachment: ${error.message}`);
    }
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!input.trim() && attachments.length === 0) return;
    if (budgetCapBlock.blocked) {
      setSendError(budgetCapBlock.detail);
      setShowBudgetSelector(true);
      return;
    }
    if (composerDisabled) return;

    // Collect attachment IDs to send with the message
    const attachmentIds = attachments.map(a => a.attachment_id);
    const submittedInput = input;
    const submittedAttachments = attachments;

    // Send message with attachment IDs (context built server-side)
    setInput('');
    setAttachments([]);
    setSendError(null);

    try {
      await onSendMessage(submittedInput, attachmentIds, submittedAttachments);
    } catch (error) {
      if (error?.status === 409) {
        setInput(submittedInput);
        setAttachments(submittedAttachments);
        setSendError(error.message || budgetCapBlock.detail);
        setShowBudgetSelector(true);
      }
    }

    // Reset textarea height after sending
    if (textareaRef.current) {
      textareaRef.current.style.height = '44px';
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    // Only hide if leaving the container (not entering a child)
    if (e.currentTarget.contains(e.relatedTarget)) return;
    setIsDragging(false);
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const droppedFiles = Array.from(e.dataTransfer.files || []);
    if (droppedFiles.length === 0) return;
    if (isUpdatingPrivacy) return;

    const MAX_FILE_SIZE = 50 * 1024 * 1024;
    const MAX_FILES = 10;

    const oversized = droppedFiles.filter(f => f.size > MAX_FILE_SIZE);
    if (oversized.length > 0) {
      alert(`Some files exceed 50MB limit: ${oversized.map(f => f.name).join(', ')}`);
      return;
    }
    if (attachments.length + droppedFiles.length > MAX_FILES) {
      alert(`Maximum ${MAX_FILES} files allowed`);
      return;
    }

    setIsUploading(true);
    try {
      for (const file of droppedFiles) {
        const result = await api.uploadAttachment(file, effectiveZdr);
        setAttachments(prev => [...prev, {
          attachment_id: result.attachment_id,
          filename: result.filename,
          status: result.status,
          warning: result.warning,
          stats: result.stats,
          cached: result.cached,
          mime_type: file.type
        }]);
      }
    } catch (error) {
      alert(error.message);
    } finally {
      setIsUploading(false);
    }
  };

  // Edit & Regenerate handlers
  const handleEditStart = (index, content) => {
    setEditingIndex(index);
    setEditingContent(content);
  };

  const handleEditCancel = () => {
    setEditingIndex(-1);
    setEditingContent('');
  };

  const handleEditSubmit = async () => {
    if (!editingContent.trim()) return;
    if (budgetCapBlock.blocked) {
      setSendError(budgetCapBlock.detail);
      setShowBudgetSelector(true);
      return;
    }

    const submittedContent = editingContent;
    const submittedIndex = editingIndex;
    setEditingIndex(-1);
    setEditingContent('');

    try {
      await onSendMessage(submittedContent, [], [], submittedIndex);
    } catch (error) {
      if (error?.status === 409) {
        setEditingIndex(submittedIndex);
        setEditingContent(submittedContent);
        setSendError(error.message || budgetCapBlock.detail);
        setShowBudgetSelector(true);
      }
    }
  };

  const handleExport = () => {
    if (!conversation) return;
    const { title, messages, created_at, metadata, total_cost } = conversation;

    // Robust mode detection
    const hasCouncilData = (messages ?? []).some(m => m.stage1 || m.stage2 || m.stage3);
    const isCouncilMode = (metadata?.council_models?.length > 0) || hasCouncilData;

    // Date formatting (single object, labeled UTC)
    const date = new Date(created_at);
    const localDate = date.toLocaleString(undefined, {
      year: 'numeric', month: 'long', day: 'numeric',
      hour: '2-digit', minute: '2-digit', timeZoneName: 'short'
    });
    const isoDate = date.toISOString();

    // Build header
    let md = `# ${title || 'Untitled Conversation'}\n`;
    md += `Date: ${localDate} (UTC: ${isoDate})\n\n`;

    // Council mode: add model list
    if (isCouncilMode && metadata?.council_models?.length > 0) {
      const councilNames = metadata.council_models.map(m => m.split('/')[1] || m).join(', ');
      md += `**Council**: ${councilNames}\n`;
      if (metadata?.chairman_model) {
        const chairName = metadata.chairman_model.split('/')[1] || metadata.chairman_model;
        md += `**Chairman**: ${chairName}\n`;
      }
      md += '\n';
    }

    md += `---\n\n`;

    // Filter to user + assistant only
    const exportMessages = (messages ?? []).filter(m => m.role === 'user' || m.role === 'assistant');

    exportMessages.forEach(msg => {
      if (msg.role === 'user') {
        md += `## User\n\n${msg.content}\n\n---\n\n`;
      } else {
        // Assistant message
        const hasStages = msg.stage1 || msg.stage2 || msg.stage3;

        if (hasStages) {
          md += `## AI Advisory Board\n\n`;

          // Stage 1: Full responses
          if (msg.stage1 && msg.stage1.length > 0) {
            md += `### Stage 1: Individual Responses\n\n`;
            msg.stage1.forEach(r => {
              const modelName = r.model?.split('/')[1] || r.model || 'Unknown';
              md += `**${modelName}**\n${r.response}\n\n`;
            });
          }

          // Stage 2: Peer rankings
          if (msg.stage2 && msg.stage2.length > 0) {
            md += `### Stage 2: Peer Rankings\n\n`;
            const labelToModel = msg.metadata?.label_to_model || {};

            msg.stage2.forEach(rank => {
              const evaluatorName = rank.model?.split('/')[1] || rank.model || 'Unknown';
              md += `**Evaluator: ${evaluatorName}**\n${rank.ranking}\n`;

              // Parsed ranking (with fallback to raw labels)
              if (rank.parsed_ranking && rank.parsed_ranking.length > 0) {
                const parsed = rank.parsed_ranking.map((label, i) => {
                  const modelName = labelToModel[label]
                    ? (labelToModel[label].split('/')[1] || labelToModel[label])
                    : label;
                  return `${i + 1}. ${modelName}`;
                }).join(', ');
                md += `Extracted: ${parsed}\n`;
              }
              md += '\n';
            });

            // Aggregate rankings table
            const aggRankings = msg.metadata?.aggregate_rankings;
            if (aggRankings && aggRankings.length > 0) {
              md += `| Rank | Model | Avg | Votes |\n`;
              md += `|------|-------|-----|-------|\n`;
              aggRankings.forEach((agg, i) => {
                const modelName = agg.model?.split('/')[1] || agg.model || 'Unknown';
                md += `| ${i + 1} | ${modelName} | ${agg.average_rank?.toFixed(2) || 'N/A'} | ${agg.rankings_count || 0} |\n`;
              });
              md += '\n';
            }
          }

          // Stage 3: Final answer
          if (msg.stage3) {
            md += `### Stage 3: Final Answer\n`;
            const chairName = msg.stage3.model?.split('/')[1] || msg.stage3.model || 'Chairman';
            md += `**Chairman**: ${chairName}`;
            if (msg.stage3.confidence) {
              md += ` | **Confidence**: ${msg.stage3.confidence}`;
            }
            md += `\n\n${msg.stage3.response}\n\n`;
          }
        } else {
          // Chat mode: simple format
          md += `## Assistant\n\n${msg.content}\n\n`;
        }

        // Turn cost (omit if missing)
        if (msg.running_cost != null && msg.running_cost > 0) {
          md += `*Turn Cost: $${msg.running_cost.toFixed(4)}*\n\n`;
        }

        md += `---\n\n`;
      }
    });

    // Total cost (omit if missing)
    if (total_cost != null && total_cost > 0) {
      md += `**Total Session Cost: $${total_cost.toFixed(4)}**\n\n`;
    }

    // Footer note about excluded messages
    md += `*Note: System messages excluded from export.*\n`;

    // Download
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${(title || 'conversation').replace(/[^a-z0-9\u4e00-\u9fff]/gi, '_').toLowerCase()}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };


  // Get council model count from conversation metadata
  const getCouncilModelCount = () => {
    if (!conversation?.metadata?.council_models) return null;
    return conversation.metadata.council_models.length;
  };

  if (!conversation) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
        <h2 className="text-2xl font-semibold mb-2">Welcome to AI Advisory Board</h2>
        <p>Create a new conversation to get started</p>
      </div>
    );
  }

  const councilCount = getCouncilModelCount();

  return (
    <div
      className="flex flex-col h-full bg-background relative"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Drag overlay */}
      {isDragging && (
        <div className="absolute inset-0 z-50 bg-primary/10 backdrop-blur-sm border-2 border-dashed border-primary/50 rounded-lg flex items-center justify-center pointer-events-none">
          <div className="text-center">
            <Paperclip className="h-10 w-10 text-primary mx-auto mb-2" />
            <p className="text-lg font-medium text-primary">Drop files here</p>
            <p className="text-sm text-muted-foreground">PDF, DOCX, images, and more</p>
          </div>
        </div>
      )}
      <div className="flex items-center justify-between p-4 border-b h-14 shrink-0 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 z-10">
        <h3 className="font-semibold truncate max-w-[60%]">{conversation.title}</h3>
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="sm" onClick={handleExport} title="Export to Markdown">
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
        </div>
      </div>


      <ScrollArea
        className="flex-1"
        viewportRef={viewportRef}
        onScroll={handleScroll}
      >
        <div className={getChatSurfaceClass('messages')}>
          {conversation.messages.length === 0 ? (
            <div className="text-center text-muted-foreground py-10">
              <h2 className="text-xl font-semibold mb-2">Start a conversation</h2>
              <p>Ask a question to consult the AI Advisory Board</p>
            </div>
          ) : (
            conversation.messages.map((msg, index) => (
              <div key={`${conversation.id}-msg-${index}-${msg.role}`} className={cn("flex flex-col gap-2", msg.role === 'user' ? "items-end" : "items-start")}>
                <div className={cn("text-xs text-muted-foreground", msg.role === 'user' ? "text-right" : "text-left")}>
                  {msg.role === 'user' ? 'You' : 'AI Advisory Board'}
                </div>

                {msg.role === 'user' ? (
                  editingIndex === index ? (
                    // Inline editor for Edit & Regenerate
                    <Card className="w-full max-w-[min(100%,42rem)] border-primary/30 bg-primary/5 p-3">
                      <Textarea
                        value={editingContent}
                        onChange={(e) => setEditingContent(e.target.value)}
                        className="min-h-[60px] max-h-[200px] resize-y text-sm mb-2"
                        autoFocus
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleEditSubmit();
                          }
                          if (e.key === 'Escape') handleEditCancel();
                        }}
                      />
                      <div className="flex flex-wrap justify-end gap-2">
                        <Button variant="ghost" size="sm" onClick={handleEditCancel}>Cancel</Button>
                        <Button size="sm" onClick={handleEditSubmit} disabled={!editingContent.trim()}>Submit</Button>
                      </div>
                    </Card>
                  ) : (
                    <div className="group/msg relative max-w-[min(85%,42rem)]">
                      <Card className="bg-primary text-primary-foreground p-3">
                        <div className="prose prose-invert max-w-none text-sm">
                          <MarkdownRenderer>{msg.content}</MarkdownRenderer>
                        </div>
                        {msg.attachments && msg.attachments.length > 0 && (
                          <div className="mt-2 pt-2 border-t border-primary-foreground/20">
                            <AttachmentPillList attachments={msg.attachments} />
                          </div>
                        )}
                      </Card>
                      {!isLoading && (
                        <button
                          onClick={() => handleEditStart(index, msg.content)}
                          className="absolute -left-8 top-1/2 rounded p-1 opacity-0 transition-opacity hover:bg-muted group-hover/msg:opacity-100"
                          title="Edit & Regenerate"
                        >
                          <Pencil className="h-3.5 w-3.5 text-muted-foreground" />
                        </button>
                      )}
                    </div>
                  )
                ) : (
                  <Card className="w-full max-w-full bg-muted/50 p-3 sm:p-4">
                    <div className="flex flex-col gap-4">
                      {/* Stage 1 Loading */}
                      {msg.loading?.stage1 && (
                        <StageProgress
                          stage="Stage 1: Individual Responses"
                          description="Each council member is providing their perspective..."
                          modelCount={councilCount}
                          icon={Users}
                        />
                      )}

                      {msg.stage1 && <Stage1 responses={msg.stage1} />}

                      {/* Stage 2 Loading */}
                      {msg.loading?.stage2 && (
                        <StageProgress
                          stage="Stage 2: Peer Ranking"
                          description="Council members are evaluating each other's responses..."
                          modelCount={councilCount}
                          icon={User}
                        />
                      )}
                      {msg.stage2 && (
                        <Stage2
                          rankings={msg.stage2}
                          labelToModel={msg.metadata?.label_to_model}
                          aggregateRankings={msg.metadata?.aggregate_rankings}
                        />
                      )}

                      {/* Stage 3 Loading */}
                      {msg.loading?.stage3 && (
                        <StageProgress
                          stage="Stage 3: Final Synthesis"
                          description="The Chairman is synthesizing the final answer..."
                          modelCount={1}
                          icon={Crown}
                        />
                      )}
                      {msg.stage3 && <Stage3 finalResponse={msg.stage3} />}

                      {/* Chat Mode */}
                      {msg.loading?.chat && !msg.content && (
                        <div className="flex flex-col gap-3 rounded-lg border border-primary/20 bg-muted/50 p-3 sm:flex-row sm:items-center">
                          <div className="relative shrink-0">
                            <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                              <Crown className="h-5 w-5 text-primary" />
                            </div>
                            <div className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-primary flex items-center justify-center">
                              <Loader2 className="h-3 w-3 animate-spin text-primary-foreground" />
                            </div>
                          </div>
                          <div className="min-w-0 flex-1">
                            <div className="font-medium text-sm">Chairman is thinking...</div>
                            <div className="text-xs text-muted-foreground">Generating response with context</div>
                          </div>
                          <div className="flex gap-1 sm:shrink-0">
                            {[...Array(3)].map((_, i) => (
                              <div
                                key={i}
                                className="w-2 h-2 rounded-full bg-primary animate-pulse"
                                style={{ animationDelay: `${i * 0.2}s` }}
                              />
                            ))}
                          </div>
                        </div>
                      )}

                      <ChainOfThought reasoning={msg.reasoning} />

                      {msg.content && (
                        <div className="prose max-w-none text-sm dark:prose-invert">
                          <MarkdownRenderer>{msg.content}</MarkdownRenderer>
                        </div>
                      )}

                      {/* Running Cost Display */}
                      {msg.role === 'assistant' && (
                        <div className="mt-2 flex flex-wrap items-center justify-between gap-2 border-t pt-2 text-xs text-muted-foreground">
                          <span>
                            Turn Cost: <span className="font-mono">${(msg.running_cost || 0).toFixed(6)}</span>
                          </span>
                          {msg.stage3?.confidence && (
                            <span className={cn(
                              "px-2 py-0.5 rounded text-xs font-medium",
                              msg.stage3.confidence === 'HIGH' && "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
                              msg.stage3.confidence === 'MEDIUM' && "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400",
                              msg.stage3.confidence === 'LOW' && "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"
                            )}>
                              {msg.stage3.confidence} Confidence
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </Card>
                )}
              </div>
            ))
          )}
          {isLoading && (
            <div className="flex justify-start">
              <Skeleton className="h-10 w-32 rounded-full" />
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>

      <div className="border-t bg-background">
        <div className={getChatSurfaceClass('composer')}>
          <TrustRow
            conversation={conversation}
            settings={settings}
            attachmentCount={attachments.length}
            budgetWarning={budgetWarning}
            onOpenBudget={() => setShowBudgetSelector(true)}
            onToggleWebSearch={() => updateSettings({ webSearchEnabled: !settings.webSearchEnabled })}
            onToggleWebDepth={() => updateSettings({ webSearchDepth: settings.webSearchDepth === 'fast' ? 'deep' : 'fast' })}
            onOpenAdvancedSettings={() => setShowAdvancedSettings(true)}
            privacyDisabled={Boolean(privacyDisabledReason)}
            privacyDisabledReason={privacyDisabledReason}
            onUpdateConversationPrivacy={async (nextZdr) => {
              if (privacyDisabledReason) return;
              setIsUpdatingPrivacy(true);
              try {
                await onUpdateConversationPrivacy?.(nextZdr);
              } catch (error) {
                alert(`Failed to update conversation privacy: ${error.message || 'Unknown error'}`);
              } finally {
                setIsUpdatingPrivacy(false);
              }
            }}
          />

          {sendError && (
            <div
              id="send-error"
              role="alert"
              className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive"
            >
              {sendError}
            </div>
          )}

          {/* Show uploaded attachments as pills */}
          {attachments.length > 0 && (
            <div className="mb-2">
              <AttachmentPillList
                attachments={attachments}
                onRemove={removeAttachment}
                onUpdate={(index, updated) => {
                  setAttachments(prev => prev.map((att, i) => i === index ? updated : att));
                }}
                showRemove={true}
                showEnhance={true}
                enhanceDisabled={isUpdatingPrivacy}
                effectiveZdr={effectiveZdr}
              />
            </div>
          )}

          <div className="relative flex items-end gap-2">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              className="hidden"
              accept=".pdf,.docx,.pptx,.xlsx,.csv,.txt,.md,.html,.json,image/*"
              multiple
            />
            <Button
              variant="outline"
              size="icon"
              className="h-10 w-10 shrink-0"
              onClick={() => fileInputRef.current?.click()}
              disabled={composerDisabled}
              title={budgetCapBlock.blocked ? budgetCapBlock.detail : 'Attach files'}
            >
              <Paperclip className="h-4 w-4" />
            </Button>

            <div className="relative flex-1">
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask your question... (Shift+Enter for new line)"
                className="min-h-[44px] max-h-[min(32vh,200px)] resize-none py-3 pr-10"
                disabled={composerDisabled}
                aria-describedby={sendError ? 'send-error' : undefined}
                rows={1}
                style={{ height: 'auto', minHeight: '44px' }}
                onInput={(e) => {
                  e.target.style.height = 'auto';
                  e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
                }}
              />
            </div>

            <Button
              onClick={(e) => handleSubmit(e)}
              disabled={(!input.trim() && attachments.length === 0) || composerDisabled}
              className="h-10 w-10 shrink-0"
              size="icon"
              title={budgetCapBlock.blocked ? budgetCapBlock.action : 'Send message'}
            >
              {isUploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            </Button>
          </div>

          <SessionBudgetSelector
            isOpen={showBudgetSelector}
            onClose={() => setShowBudgetSelector(false)}
            onConfirm={handleBudgetConfirm}
            currentBudget={sessionBudget}
          />
          <AdvancedSettingsPanel
            isOpen={showAdvancedSettings}
            onClose={() => setShowAdvancedSettings(false)}
            settings={settings}
            onSave={updateSettings}
            nextMessageMode={nextMessageMode}
          />
        </div>
      </div>
    </div>
  );
}
