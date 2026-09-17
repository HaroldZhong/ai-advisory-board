import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import { streamReducer } from '../utils/streamReducer';
import { applyStreamUpdateToActiveConversation } from '../utils/reasoningMessages';
import { rollbackFailedSendConversation, reconcileInterruptedRun } from '../utils/optimisticMessages';
import { normalizeAdvancedSettingsForMode } from '../utils/advancedSettingsAvailability';
import { resolveSendMode } from '../utils/modePrediction';
import { resolveEffectiveZdr } from '../utils/trustState';
import { toast } from './use-toast';
import { formatStreamErrorMessage } from '../utils/streamErrors';

/**
 * Owns the SSE turn-event pipeline (backend/turn_pipeline.py, audit §9):
 * dispatches every stream event through the pure `streamReducer`, applies
 * the resulting state to the active conversation, and handles the
 * side effects the old inline switch in App.jsx used to trigger directly
 * (reloading the conversation list on title_complete/complete, toasting on
 * error, rolling back optimistic messages on network failure).
 *
 * Conversation/budget-warning state is still owned by the caller (App.jsx)
 * because it's also mutated by non-streaming handlers (privacy, thinking
 * effort, session policy) — this hook reads/writes it via the passed-in
 * setter rather than duplicating the state.
 *
 * `conversationId` must be the route param (App.jsx's useParams().conversationId),
 * not `currentConversation?.id`: currentConversation is loaded asynchronously
 * after navigation and is briefly stale on conversation switch, which would
 * otherwise send the turn to the previous conversation.
 */
export function useStreamingConversation({
  conversationId,
  currentConversation,
  setCurrentConversation,
  setBudgetWarning,
  availableModels,
  loadConversations,
  settings,
  onRetainDraft,
  zdrAvailable = true,
}) {
  const [isLoading, setIsLoading] = useState(false);
  const activeRequest = useRef(null);
  const latestRequest = useRef(null);
  const [streamStatus, setStreamStatus] = useState(null);

  useEffect(() => () => {
    activeRequest.current?.controller.abort();
    latestRequest.current = null;
  }, []);

  const stopMessage = () => {
    const request = activeRequest.current;
    if (!request) return;
    setStreamStatus({ conversationId: request.conversationId, text: 'Stop requested' });
    request.controller.abort();
  };

  // Warn before an accidental tab close/reload mid-stream (P3-T8 item 5) —
  // standard browser confirm dialog, only attached while a turn is in
  // flight. The desktop WebView shell has no navigation chrome to trigger
  // this from, so it's a harmless no-op there.
  useEffect(() => {
    if (!isLoading) return undefined;

    const handleBeforeUnload = (event) => {
      event.preventDefault();
      event.returnValue = '';
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [isLoading]);

  const sendMessage = async (content, attachmentIds = [], attachmentMetadata = [], editIndex = -1, options = {}) => {
    const { mode: explicitMode } = options;
    const targetConversationId = conversationId;
    if (!targetConversationId || activeRequest.current) return;
    const request = {
      id: crypto.randomUUID(), controller: new AbortController(),
      conversationId: targetConversationId, runId: null,
    };
    activeRequest.current = request;
    latestRequest.current = request;
    let sawEvent = false;
    let terminal = false;
    let streamError = null;
    // React may apply queued updates after the transport's finally has run.
    const isCurrent = () => latestRequest.current === request;

    const previousMessages = editIndex >= 0
      ? [...(currentConversation?.messages || [])]
      : null;
    const updateTargetConversation = (updater) => {
      setCurrentConversation((prev) => (
        isCurrent() ? applyStreamUpdateToActiveConversation(prev, targetConversationId, updater) : prev
      ));
    };

    setIsLoading(true);
    setStreamStatus(null);
    try {
      const userMessage = {
        role: 'user',
        content,
        attachments: attachmentMetadata,
      };

      // Edit & Regenerate: truncate local state to edit point
      if (editIndex >= 0) {
        updateTargetConversation((prev) => ({
          ...prev,
          messages: [...prev.messages.slice(0, editIndex), userMessage],
        }));
      } else {
        updateTargetConversation((prev) => ({
          ...prev,
          messages: [...prev.messages, userMessage],
        }));
      }

      // Routing is backend-owned: without an explicit override the request
      // carries mode "auto" and prepare_turn resolves it edit-aware. An
      // armed "Ask the council" send (P3-T4) passes an explicit mode that
      // wins on both the wire and the optimistic skeleton below, mirroring
      // prepare_turn's "explicit request.mode wins" rule.
      const predictedMode = resolveSendMode(explicitMode, {
        messageCount: currentConversation.messages.length,
        editIndex,
        defaultMode: currentConversation?.metadata?.default_mode,
      });
      const requestSettings = normalizeAdvancedSettingsForMode(settings, predictedMode);
      requestSettings.zdrEnabled = resolveEffectiveZdr(currentConversation, settings, zdrAvailable);

      if (predictedMode === 'council') {
        const assistantMessage = {
          role: 'assistant',
          client_request_id: request.id,
          stage1: null,
          stage2: null,
          stage3: null,
          metadata: null,
          loading: {
            stage1: false,
            stage2: false,
            stage3: false,
            stage3_status: 'pending',
          },
        };
        request.draft = assistantMessage;

        updateTargetConversation((prev) => ({
          ...prev,
          messages: [...prev.messages, assistantMessage],
        }));
      } else {
        const assistantMessage = {
          role: 'assistant',
          client_request_id: request.id,
          content: '',
          loading: {
            chat: true,
          },
        };
        request.draft = assistantMessage;

        updateTargetConversation((prev) => ({
          ...prev,
          messages: [...prev.messages, assistantMessage],
        }));
      }

      const knownEventTypes = new Set([
        'stage1_start', 'stage1_model_complete', 'stage1_complete',
        'stage2_start', 'stage2_complete',
        'stage3_start', 'stage3_complete',
        'chat_start', 'chat_response',
        'reasoning_delta', 'content_delta',
        'title_complete', 'budget_warning',
        'complete', 'error', 'turn_state',
      ]);

      await api.sendMessageStream(targetConversationId, content, (eventType, event) => {
        if (!isCurrent() || request.controller.signal.aborted) return;
        sawEvent = true;
        if (eventType === 'turn_state') request.runId = event.data.run_id;
        if (eventType === 'complete' || eventType === 'error') terminal = true;
        if (!knownEventTypes.has(eventType)) {
          console.warn('Unknown event type:', eventType);
          return;
        }

        if (eventType === 'title_complete') {
          loadConversations();
          return;
        }

        request.draft = streamReducer(
          { conversation: { messages: [request.draft] }, isLoading: true, budgetWarning: null },
          event, { availableModels },
        ).conversation.messages[0];
        updateTargetConversation((prev) => {
          const result = streamReducer(
            { conversation: prev, isLoading: true, budgetWarning: null },
            event,
            { availableModels },
          );
          if (result.budgetWarning !== null) {
            setBudgetWarning(result.budgetWarning);
          }
          return result.conversation;
        });

        if (eventType === 'complete') {
          loadConversations();
        } else if (eventType === 'error') {
          streamError = new Error(formatStreamErrorMessage(event.message));
        }
      }, explicitMode || 'auto', attachmentIds, {
        enabled: settings.webSearchEnabled,
        evidenceSourceIds: options.evidenceSourceIds,
        expectedCouncilModels: options.expectedCouncilModels,
        expectedChairmanModel: options.expectedChairmanModel,
        depth: settings.webSearchDepth,
        customInstructions: requestSettings.customInstructions,
        zdrEnabled: requestSettings.zdrEnabled,
        executionMode: requestSettings.executionMode,
        ragPreset: requestSettings.ragPreset,
        modelTier: requestSettings.modelTier,
      }, editIndex, request.controller.signal);
      if (streamError) throw streamError;
      if (!terminal) throw new Error('Connection lost before the final turn state');
    } catch (error) {
      if (!isCurrent()) return;
      const stopped = request.controller.signal.aborted || error?.name === 'AbortError';
      const isPreflightRejection = !sawEvent && [400, 403, 409, 412].includes(error?.status);
      if (stopped || sawEvent || error?.status == null) {
        const reason = stopped ? 'Stop requested' : (streamError ? 'Response failed' : 'Connection lost');
        setStreamStatus({ conversationId: targetConversationId, text: reason });
        const scope = { conversationId: targetConversationId, requestId: request.id, runId: request.runId, reason };
        updateTargetConversation((prev) => reconcileInterruptedRun(prev, null, scope));
        let persisted = null;
        try {
          persisted = await api.getConversation(targetConversationId, { signal: AbortSignal.timeout(5000) });
          updateTargetConversation((prev) => reconcileInterruptedRun(prev, persisted, scope));
          if (isCurrent()) loadConversations();
        } catch {
          // Keep the visible draft. A failed read cannot prove it was saved.
        }
        if (isCurrent() && request.draft) {
          const draft = reconcileInterruptedRun(
            { id: targetConversationId, messages: [request.draft] }, persisted, scope,
          ).messages[0];
          if (draft.unconfirmed) onRetainDraft?.({ id: request.id, conversationId: targetConversationId, prompt: content, message: draft });
        }
        if (!stopped) {
          toast({ variant: 'destructive', title: reason, description: error.message });
        }
      } else {
        if (!isPreflightRejection) {
          toast({ variant: 'destructive', title: 'Response failed', description: error.message || 'Unknown error' });
        }
        updateTargetConversation((prev) => rollbackFailedSendConversation(prev, {
          conversationId: targetConversationId, editIndex, previousMessages,
        }));
      }
      if (isPreflightRejection) throw error;
    } finally {
      if (isCurrent()) {
        activeRequest.current = null;
        setIsLoading(false);
      }
    }
  };

  return { sendMessage, stopMessage, isLoading, streamStatus };
}
