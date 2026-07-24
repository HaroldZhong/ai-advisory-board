import { useState, useEffect, useRef } from 'react';
import MarkdownRenderer from './MarkdownRenderer';
import Stage1 from './Stage1';
import Stage2 from './Stage2';
import Stage3 from './Stage3';
import ReasoningSection from './ReasoningSection';
import SessionBudgetSelector from './SessionBudgetSelector';
import AdvancedSettingsPanel from './AdvancedSettingsPanel';
import { api } from '../api';
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Paperclip, Send, Download, Loader2, Users, User, Crown, Pencil } from "lucide-react";
import { cn } from "@/lib/utils";
import AttachmentPill, { AttachmentPillList } from './AttachmentPill';
import { useSettings } from '@/contexts/SettingsContext';
import TrustRow from './TrustRow';
import { getChatSurfaceClass } from '@/utils/responsiveChatLayout';
import { buildBudgetPolicyUpdate, formatCurrency, getBudgetCapBlockState, getPrivacyToggleDisabledReason, resolveEffectiveZdr } from '@/utils/trustState';
import { predictNextMessageMode } from '../utils/modePrediction';
import { extractMessageAttachmentIds } from '../utils/messageAttachments';
import { toast } from '@/hooks/use-toast';
import { getExportSavedDescription } from '@/utils/conversationExport';

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
  onUpdateThinkingEffort,
  budgetWarning,
  isLoading,
  zdrAvailable = true,
}) {
  const [input, setInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isUpdatingPrivacy, setIsUpdatingPrivacy] = useState(false);
  const [isUpdatingThinkingEffort, setIsUpdatingThinkingEffort] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [attachments, setAttachments] = useState([]);  // Uploaded attachment metadata
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  // D3: guards against a double-dispatch while the pre-send estimate is in flight.
  // handleSubmit awaits getTurnEstimate before clearing the input / setting any
  // disabled state, so without this a double-click could start two sends (Codex #110).
  const estimatingRef = useRef(false);

  const [showBudgetSelector, setShowBudgetSelector] = useState(false);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [sendError, setSendError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [editingIndex, setEditingIndex] = useState(-1);
  const [editingContent, setEditingContent] = useState('');
  const [editingAttachmentIds, setEditingAttachmentIds] = useState([]);
  const [editingAttachmentMetadata, setEditingAttachmentMetadata] = useState([]);
  const [askCouncil, setAskCouncil] = useState(false);
  const [showCouncilConfirm, setShowCouncilConfirm] = useState(false);
  // v1.3.0 D3: approximate pre-send cost estimate shown in the council confirm.
  const [turnEstimate, setTurnEstimate] = useState(null);
  // v1.3.0 D3: true while the pre-send estimate is in flight -- disables the composer
  // so edits can't be lost when a slow estimate resolves (Codex #110).
  const [isEstimating, setIsEstimating] = useState(false);
  const { settings, updateSettings } = useSettings();

  const sessionPolicy = conversation?.session_policy || {};
  const sessionBudget = sessionPolicy.budget_usd ?? null;
  // Display-only prediction; routing itself is backend-owned (mode "auto").
  const nextMessageMode = predictNextMessageMode({
    messageCount: conversation?.messages?.length || 0,
    defaultMode: conversation?.metadata?.default_mode,
  });
  // "Ask the council" (P3-T4) only makes sense when the next send would
  // otherwise run chat — i.e. a default_mode="chat" conversation, or a
  // legacy/council-default conversation mid-thread (past its first turn).
  const canAskCouncil = nextMessageMode === 'chat';
  const effectiveZdr = resolveEffectiveZdr(conversation, settings, zdrAvailable);
  const budgetCapBlock = getBudgetCapBlockState(conversation);
  // Granular busy flags (P3-T8 item 2): a single isLoading used to disable
  // the whole composer area, including controls with no real conflict with
  // an in-flight stream. Only sending a NEW turn (and its textarea/attach
  // toggle) needs to wait on isLoading — thinking effort tracks its own busy
  // state and carries no per-turn promise, so it isn't blocked just because
  // a response happens to be streaming. The attach button DOES still need
  // isUploading (Codex review, round 2 item 2): handleFileUpload validates
  // the 10-file cap against a stale `attachments.length` snapshot taken at
  // call time, so a second upload starting before the first one's state
  // updates land could slip past the cap. Privacy is DELIBERATELY still
  // blocked by isLoading (Codex review, round 4 — see
  // getPrivacyToggleDisabledReason's isStreaming param): prepare_turn
  // resolves zdr_enabled once per turn, so flipping the toggle mid-stream
  // would show "ZDR enforced" while the in-flight turn keeps its captured
  // (possibly non-ZDR) routing.
  const sendDisabled = isLoading || isUploading || isUpdatingPrivacy || isUpdatingThinkingEffort || isEstimating || budgetCapBlock.blocked;
  // D3: freeze the composer content (attach + remove) while an estimate is in flight
  // or a confirm is pending, so what the estimate was computed on is exactly what
  // gets sent -- no add/remove/edit can slip in between (Codex #110).
  const attachDisabled = isUploading || isUpdatingPrivacy || isEstimating || showCouncilConfirm || budgetCapBlock.blocked;
  const composerDisabled = sendDisabled;
  const privacyDisabledReason = getPrivacyToggleDisabledReason({
    isStreaming: isLoading,
    isUploading,
    isUpdatingPrivacy,
    // Only the ENABLE direction is blocked off-provider — an explicit
    // conversation-level ZDR (effectiveZdr already true) must still be
    // disable-able so the user can consciously turn it off.
    isEnablingUnavailable: !zdrAvailable && !effectiveZdr,
  });
  const thinkingDisabledReason = isUpdatingThinkingEffort
    ? 'Thinking effort update is being saved'
    : null;

  const handleBudgetConfirm = async (budgetUsd, allowOverage = true) => {
    try {
      setSendError(null);
      // v1.3.0 D3: carry the hard-cap opt-in so users can enforce the 409 cap.
      await onUpdateSessionPolicy?.(buildBudgetPolicyUpdate(budgetUsd, allowOverage));
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

  // ChatInterface is not remounted on a conversation switch (App renders it without
  // a key), so its composer state survives. Reset the council-confirm state when the
  // conversation changes, or a stale confirm from conversation A -- including its
  // per-conversation cost estimate -- would render over conversation B and dispatch
  // to B while showing A's number (Codex #110 audit P3).
  useEffect(() => {
    setAskCouncil(false);
    setShowCouncilConfirm(false);
    setTurnEstimate(null);
  }, [conversation?.id]);

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
    // D3: don't let a removal race a pending estimate/confirm -- the send is bound to
    // the attachments the estimate was computed on (Codex #110).
    if (isEstimating || showCouncilConfirm) return;
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
    // A pre-send estimate is already in flight for this composer -- ignore the repeat
    // click/Enter so a double-tap during the await can't start two paid sends (Codex #110).
    if (estimatingRef.current) return;

    // Snapshot the composer BEFORE any await so a slow estimate can't send stale content
    // or let the later setInput('') wipe edits made during the await (Codex #110).
    const submittedInput = input;
    const submittedAttachments = attachments;
    const attachmentIds = attachments.map(a => a.attachment_id);

    // D3 (§5.1) soft seatbelt: warn with an APPROXIMATE pre-send estimate before a
    // LARGE predicted turn -- ANY mode -- and before an explicitly armed council send
    // (P3-T4). Covers auto council-first sends (Codex #110) and large chat turns with
    // an expensive chairman (Codex #110 R2), not only the manual "Ask the council"
    // override. Resolve the actual next mode: armed -> council; else the auto
    // prediction (council on a council-default first turn, otherwise chat).
    const resolvedSendMode = askCouncil ? 'council' : nextMessageMode;
    if (!showCouncilConfirm) {
      // Fetch the estimate for the mode that will actually run. Never blocks: a
      // failed estimate resolves to null and the send proceeds. estimatingRef guards
      // re-entry synchronously; isEstimating disables the composer so edits can't be lost.
      let estimate = null;
      if (conversation?.id) {
        estimatingRef.current = true;
        setIsEstimating(true);
        try {
          estimate = await api.getTurnEstimate(conversation.id, {
            content: submittedInput,
            hasAttachments: submittedAttachments.length > 0,
            mode: resolvedSendMode,
            executionMode: settings.executionMode,
            ragPreset: settings.ragPreset,
            modelTier: settings.modelTier,
          });
        } catch {
          estimate = null;
        } finally {
          estimatingRef.current = false;
          setIsEstimating(false);
        }
      }
      // Confirm before an explicitly armed council send (P3-T4) or any LARGE predicted
      // turn (D3). A turn whose estimate is known and not large dispatches uninterrupted.
      if (askCouncil || estimate?.is_large) {
        setTurnEstimate(estimate);
        setShowCouncilConfirm(true);
        return;
      }
    }

    const sendOptions = askCouncil ? { mode: 'council' } : {};

    // Send message with attachment IDs (context built server-side)
    setInput('');
    setAttachments([]);
    setSendError(null);
    setAskCouncil(false);
    setShowCouncilConfirm(false);
    setTurnEstimate(null);

    try {
      await onSendMessage(submittedInput, attachmentIds, submittedAttachments, -1, sendOptions);
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

  const handleCouncilConfirmCancel = () => {
    setShowCouncilConfirm(false);
    setAskCouncil(false);
    setTurnEstimate(null);
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
  const handleEditStart = (index, message) => {
    setEditingIndex(index);
    setEditingContent(message.content);
    // Re-apply the original message's attachments on regenerate (P3-T8 item
    // 3) — the resend used to hardcode an empty attachment list here.
    setEditingAttachmentIds(extractMessageAttachmentIds(message));
    setEditingAttachmentMetadata(message.attachments || []);
  };

  const handleEditCancel = () => {
    setEditingIndex(-1);
    setEditingContent('');
    setEditingAttachmentIds([]);
    setEditingAttachmentMetadata([]);
  };

  const handleEditSubmit = async () => {
    if (!editingContent.trim()) return;
    if (budgetCapBlock.blocked) {
      setSendError(budgetCapBlock.detail);
      setShowBudgetSelector(true);
      return;
    }
    // ponytail: the D3 pre-send estimate/confirm (§5.1) covers the primary compose->send
    // path (handleSubmit). Editing/regenerating the FIRST message re-runs council but
    // dispatches here without the estimate confirm -- an accepted residual (Codex #110
    // audit P3): the edit flow carries its own saved-edit state and routing the shared
    // confirm dialog through it is disproportionate for this rare edge case. Add a
    // shared confirm gate here if edit-of-first-message warnings are later required.

    const submittedContent = editingContent;
    const submittedIndex = editingIndex;
    const submittedAttachmentIds = editingAttachmentIds;
    const submittedAttachmentMetadata = editingAttachmentMetadata;
    setEditingIndex(-1);
    setEditingContent('');
    setEditingAttachmentIds([]);
    setEditingAttachmentMetadata([]);

    try {
      await onSendMessage(submittedContent, submittedAttachmentIds, submittedAttachmentMetadata, submittedIndex);
    } catch (error) {
      if (error?.status === 409) {
        setEditingIndex(submittedIndex);
        setEditingContent(submittedContent);
        setEditingAttachmentIds(submittedAttachmentIds);
        setEditingAttachmentMetadata(submittedAttachmentMetadata);
        setSendError(error.message || budgetCapBlock.detail);
        setShowBudgetSelector(true);
      }
    }
  };

  const handleExport = async () => {
    if (!conversation?.id || isExporting) return;

    try {
      setIsExporting(true);
      const result = await api.exportConversation(conversation.id);
      toast({
        title: 'Export saved',
        description: <span className="break-all">{getExportSavedDescription(result.path)}</span>,
      });
    } catch (error) {
      console.error('Failed to export conversation', error);
      toast({
        variant: 'destructive',
        title: 'Export failed',
        description: error?.message || 'Could not save the Markdown export.',
      });
    } finally {
      setIsExporting(false);
    }
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
          <Button variant="ghost" size="sm" onClick={handleExport} disabled={isExporting} title="Export to Markdown">
            {isExporting ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Download className="mr-2 h-4 w-4" />
            )}
            {isExporting ? 'Exporting' : 'Export'}
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
            conversation.messages.map((msg, index) => {
              // No schema change: a user message doesn't record which mode it
              // triggered, so the "council" badge is derived from the message
              // that follows it having a stage3 result (P3-T4).
              const triggeredCouncil = msg.role === 'user' && Boolean(conversation.messages[index + 1]?.stage3);
              return (
              <div key={`${conversation.id}-msg-${index}-${msg.role}`} className={cn("flex flex-col gap-2", msg.role === 'user' ? "items-end" : "items-start")}>
                <div className={cn("flex items-center gap-1.5 text-xs text-muted-foreground", msg.role === 'user' ? "flex-row-reverse" : "text-left")}>
                  <span>{msg.role === 'user' ? 'You' : 'AI Advisory Board'}</span>
                  {triggeredCouncil && (
                    <span className="inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase text-muted-foreground">
                      <Users className="h-2.5 w-2.5" />
                      Council
                    </span>
                  )}
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
                          onClick={() => handleEditStart(index, msg)}
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

                      {msg.stage1 && (
                        <Stage1
                          responses={msg.stage1}
                          messageKey={`${conversation?.id || 'conversation'}-${index}`}
                          showReasoningByDefault={settings.showReasoningByDefault}
                        />
                      )}

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
                          messageKey={`${conversation?.id || 'conversation'}-${index}`}
                          showReasoningByDefault={settings.showReasoningByDefault}
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
                      {msg.stage3 && (
                        <Stage3
                          finalResponse={msg.stage3}
                          messageKey={`${conversation?.id || 'conversation'}-${index}`}
                          showReasoningByDefault={settings.showReasoningByDefault}
                        />
                      )}

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

                      <ReasoningSection
                        modelId={conversation?.metadata?.chairman_model}
                        modelLabel="Chairman"
                        reasoningText={msg.reasoning}
                        status={msg.loading?.chat ? 'streaming' : 'complete'}
                        defaultExpanded={settings.showReasoningByDefault}
                        storageKey={`aab.reasoning.${conversation?.id || 'conversation'}-${index}.chat`}
                      />

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
              );
            })
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
            // D3: don't let routing settings change while an estimate/confirm is
            // pending -- the shown estimate was computed on the current settings (Codex #110).
            onOpenAdvancedSettings={() => { if (!isEstimating && !showCouncilConfirm) setShowAdvancedSettings(true); }}
            privacyDisabled={Boolean(privacyDisabledReason)}
            privacyDisabledReason={privacyDisabledReason}
            thinkingDisabled={Boolean(thinkingDisabledReason)}
            thinkingDisabledReason={thinkingDisabledReason}
            zdrAvailable={zdrAvailable}
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
            onUpdateThinkingEffort={async (nextEffort) => {
              if (thinkingDisabledReason) return;
              setIsUpdatingThinkingEffort(true);
              try {
                await onUpdateThinkingEffort?.(nextEffort);
              } catch (error) {
                alert(`Failed to update thinking effort: ${error.message || 'Unknown error'}`);
              } finally {
                setIsUpdatingThinkingEffort(false);
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

          {canAskCouncil && (
            <div className="mb-2 flex items-center gap-2">
              <Button
                type="button"
                variant={askCouncil ? 'default' : 'outline'}
                size="sm"
                aria-pressed={askCouncil}
                disabled={composerDisabled}
                onClick={() => {
                  setAskCouncil((prev) => !prev);
                  setShowCouncilConfirm(false);
                }}
              >
                <Users className="mr-2 h-3.5 w-3.5" />
                Ask the council
              </Button>
            </div>
          )}

          {showCouncilConfirm && (
            <div
              role="alert"
              className="mb-2 flex flex-wrap items-center justify-between gap-2 rounded-md border border-primary/30 bg-primary/5 px-3 py-2 text-sm"
            >
              <span>
                {(askCouncil || nextMessageMode === 'council')
                  ? 'Council run uses every council model — costs more and takes 2–5 min.'
                  : 'This looks like a larger-than-usual turn.'}
                {turnEstimate?.predicted_cost > 0 && (
                  <> Est. ~{formatCurrency(turnEstimate.predicted_cost)} (approximate).</>
                )}
              </span>
              <div className="flex gap-2">
                <Button type="button" variant="ghost" size="sm" onClick={handleCouncilConfirmCancel}>
                  Cancel
                </Button>
                <Button type="button" size="sm" onClick={handleSubmit}>
                  Confirm
                </Button>
              </div>
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
              disabled={attachDisabled}
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
                // D3: lock the prompt while a confirm is pending so the sent turn
                // matches the estimate shown (Codex #110). isEstimating is already in
                // composerDisabled; showCouncilConfirm covers the confirm window.
                disabled={composerDisabled || showCouncilConfirm}
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
            currentAllowOverage={sessionPolicy.allow_overage ?? true}
            currentNotifyThresholds={sessionPolicy.notify_thresholds}
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
