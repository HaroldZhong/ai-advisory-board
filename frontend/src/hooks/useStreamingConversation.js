import { useEffect, useState } from 'react';
import { api } from '../api';
import { streamReducer } from '../utils/streamReducer';
import { applyStreamUpdateToActiveConversation } from '../utils/reasoningMessages';
import { rollbackFailedSendConversation } from '../utils/optimisticMessages';
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
  zdrAvailable = true,
}) {
  const [isLoading, setIsLoading] = useState(false);

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
    if (!targetConversationId) return;

    const previousMessages = editIndex >= 0
      ? [...(currentConversation?.messages || [])]
      : null;
    const updateTargetConversation = (updater) => {
      setCurrentConversation((prev) => (
        applyStreamUpdateToActiveConversation(prev, targetConversationId, updater)
      ));
    };

    setIsLoading(true);
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

        updateTargetConversation((prev) => ({
          ...prev,
          messages: [...prev.messages, assistantMessage],
        }));
      } else {
        const assistantMessage = {
          role: 'assistant',
          content: '',
          loading: {
            chat: true,
          },
        };

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
        'complete', 'error',
      ]);

      await api.sendMessageStream(targetConversationId, content, (eventType, event) => {
        if (!knownEventTypes.has(eventType)) {
          console.warn('Unknown event type:', eventType);
          return;
        }

        if (eventType === 'title_complete') {
          loadConversations();
          return;
        }

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
          setIsLoading(false);
        } else if (eventType === 'error') {
          console.error('Stream error:', event.message);
          toast({
            variant: 'destructive',
            title: 'Response failed',
            description: formatStreamErrorMessage(event.message),
          });
          api.getConversation(targetConversationId)
            .then((persisted) => {
              setCurrentConversation((prev) => (
                prev?.id === targetConversationId ? persisted : prev
              ));
              loadConversations();
            })
            .catch(() => {
              setCurrentConversation((prev) => (
                rollbackFailedSendConversation(prev, {
                  conversationId: targetConversationId,
                  editIndex,
                  previousMessages,
                })
              ));
            })
            .finally(() => {
              setIsLoading(false);
            });
        }
      }, explicitMode || 'auto', attachmentIds, {
        enabled: settings.webSearchEnabled,
        depth: settings.webSearchDepth,
        customInstructions: requestSettings.customInstructions,
        zdrEnabled: requestSettings.zdrEnabled,
        executionMode: requestSettings.executionMode,
        ragPreset: requestSettings.ragPreset,
        modelTier: requestSettings.modelTier,
      }, editIndex);
    } catch (error) {
      console.error('Failed to send message:', error);
      const isBudgetCapError = error?.status === 409;
      if (!isBudgetCapError) {
        alert(`Failed to send message: ${error.message || 'Unknown error'}`);
      }
      setCurrentConversation((prev) => {
        return rollbackFailedSendConversation(prev, {
          conversationId: targetConversationId,
          editIndex,
          previousMessages,
        });
      });
      setIsLoading(false);
      if (isBudgetCapError) {
        throw error;
      }
    }
  };

  return { sendMessage, isLoading };
}
