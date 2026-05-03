import { useState, useEffect, useRef } from 'react';
import { Routes, Route, useParams, useNavigate, useLocation } from 'react-router-dom';
import LandingPage from './landing/LandingPage';
import Sidebar, { SidebarContent } from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import { api } from './api';
import { Toaster } from "@/components/ui/toaster";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Menu } from "lucide-react";

import FirstRunSetup from './components/FirstRunSetup';
import ModelSelector from './components/ModelSelector';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import { calculateUsageCost, calculateStage1Cost, calculateStage2Cost, calculateStage3Cost } from './utils/cost';
import { SettingsProvider, useSettings } from './contexts/SettingsContext';
import { normalizeAdvancedSettingsForMode } from './utils/advancedSettingsAvailability';
import {
  buildConfigStatusFailureState,
  buildConfigStatusSuccessState,
  getConfigStatusRetryDelayMs,
} from './utils/configStatus';
import { createConversationWithDefaults } from './utils/conversationCreation';
import { shouldConsumeOneShotSignal } from './utils/oneShotSignal';
import { rollbackFailedSendConversation } from './utils/optimisticMessages';
import {
  mergeConversationThinkingEffortUpdate,
  setConversationThinkingEffortMetadata,
} from './utils/thinkingEffort';
import {
  appendContentDeltaToMessage,
  appendReasoningDeltaToMessage,
  applyStreamUpdateToActiveConversation,
  markLastAssistantStreamInterrupted,
  mergeReasoningBufferIntoResult,
  mergeReasoningBuffersIntoResults,
} from './utils/reasoningMessages';
import {
  mergeConversationPrivacyUpdate,
  resolveEffectiveZdr,
  setConversationPrivacyMetadata,
} from './utils/trustState';
import { toast } from './hooks/use-toast';

function ConversationView({
  conversations,
  onConversationsChange,
  availableModels,
  onShowAnalytics,
  showAnalytics,
  onCloseAnalytics,
  // Folder props
  folders,
  onCreateFolder,
  onRenameFolder,
  onDeleteFolder,
  // Conversation management props
  onRenameConversation,
  onDeleteConversation,
  onMoveConversation,
  openModelPickerSignal = 0,
  lastConsumedModelPickerSignal = 0,
  onConsumeModelPickerSignal,
}) {
  const { conversationId } = useParams();
  const navigate = useNavigate();

  const [currentConversation, setCurrentConversation] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isModelSelectorOpen, setIsModelSelectorOpen] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [budgetWarning, setBudgetWarning] = useState(null);
  const { settings } = useSettings();

  useEffect(() => {
    if (shouldConsumeOneShotSignal(openModelPickerSignal, lastConsumedModelPickerSignal)) {
      onConsumeModelPickerSignal?.(openModelPickerSignal);
      setIsModelSelectorOpen(true);
    }
  }, [openModelPickerSignal, lastConsumedModelPickerSignal, onConsumeModelPickerSignal]);

  // Load conversation details when URL changes
  useEffect(() => {
    if (!conversationId) {
      setCurrentConversation(null);
      return;
    }

    let cancelled = false;

    const loadConversation = async (id) => {
      try {
        const conv = await api.getConversation(id);
        if (!cancelled) {
          setCurrentConversation(conv);
        }
      } catch (error) {
        if (!cancelled) {
          console.error('Failed to load conversation:', error);
          // Navigate back to home if conversation not found
          navigate('/', { replace: true });
        }
      }
    };

    loadConversation(conversationId);

    return () => {
      cancelled = true;
    };
  }, [conversationId, navigate]);

  useEffect(() => {
    setBudgetWarning(null);
  }, [conversationId]);

  const handleNewConversation = () => {
    setIsModelSelectorOpen(true);
  };

  const handleModelConfirm = async ({
    councilMembers,
    chairmanModel,
    presetId,
    zdrEnabled,
    budgetUsd,
  }) => {
    try {
      const newConv = await createConversationWithDefaults({
        apiClient: api,
        topic: "New Conversation",
        councilMembers,
        chairmanModel,
        presetId,
        zdrEnabled,
        defaultSessionBudgetUsd: budgetUsd ?? settings.defaultSessionBudgetUsd,
      });

      // Update conversations list
      onConversationsChange([
        {
          id: newConv.id,
          title: newConv.title || "New Conversation",
          created_at: newConv.created_at,
          message_count: 0,
          folder_id: newConv.metadata?.folder_id,
        },
        ...conversations,
      ]);

      // Navigate to new conversation
      navigate(`/c/${newConv.id}`);
    } catch (error) {
      console.error('Failed to create conversation:', error);
    }
  };

  const handleSelectConversation = (id) => {
    navigate(`/c/${id}`);
  };

  const handleDeleteConversationFromSidebar = async (id) => {
    await onDeleteConversation(id);
    if (id === conversationId) {
      setCurrentConversation(null);
      navigate('/app', { replace: true });
    }
  };

  const loadConversations = async () => {
    try {
      const convs = await api.listConversations();
      onConversationsChange(convs);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  };

  const handleUpdateSessionPolicy = async (policyUpdate) => {
    if (!conversationId) return null;

    const state = await api.updateSessionPolicy(conversationId, policyUpdate);
    setCurrentConversation((prev) => prev ? ({
      ...prev,
      session_policy: state.policy,
      session_usage: state.usage,
      budget_spent_pct: state.budget_spent_pct,
    }) : prev);
    setBudgetWarning(null);
    return state;
  };

  const handleUpdateConversationPrivacy = async (zdrEnabled) => {
    if (!conversationId || isLoading) return null;

    const targetConversationId = conversationId;
    const previousZdr = currentConversation?.id === targetConversationId
      ? currentConversation?.metadata?.zdr_enabled
      : undefined;

    setCurrentConversation((prev) => (
      setConversationPrivacyMetadata(prev, targetConversationId, zdrEnabled)
    ));

    try {
      const updated = await api.updateConversation(targetConversationId, { zdr_enabled: zdrEnabled });
      setCurrentConversation((prev) => mergeConversationPrivacyUpdate(prev, updated));
      loadConversations();
      return updated;
    } catch (error) {
      setCurrentConversation((prev) => (
        setConversationPrivacyMetadata(prev, targetConversationId, previousZdr)
      ));
      throw error;
    }
  };

  const handleUpdateThinkingEffort = async (thinkingEffort) => {
    if (!conversationId || isLoading) return null;

    const targetConversationId = conversationId;
    const previousThinkingEffort = currentConversation?.id === targetConversationId
      ? currentConversation?.metadata?.thinking_effort
      : undefined;

    setCurrentConversation((prev) => (
      setConversationThinkingEffortMetadata(prev, targetConversationId, thinkingEffort)
    ));

    try {
      const updated = await api.updateConversation(targetConversationId, { thinking_effort: thinkingEffort });
      setCurrentConversation((prev) => mergeConversationThinkingEffortUpdate(prev, updated));
      loadConversations();
      return updated;
    } catch (error) {
      setCurrentConversation((prev) => (
        setConversationThinkingEffortMetadata(prev, targetConversationId, previousThinkingEffort)
      ));
      throw error;
    }
  };

  const handleSendMessage = async (content, attachmentIds = [], attachmentMetadata = [], editIndex = -1) => {
    if (!conversationId) return;

    const targetConversationId = conversationId;
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
        attachments: attachmentMetadata
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

      // Determine mode: if after truncation we're at msg index 0, it's council mode
      const effectiveMsgCount = editIndex >= 0 ? editIndex : currentConversation.messages.length;
      const isFollowUp = effectiveMsgCount > 0;
      const mode = isFollowUp ? 'chat' : 'council';
      const requestSettings = normalizeAdvancedSettingsForMode(settings, mode);
      requestSettings.zdrEnabled = resolveEffectiveZdr(currentConversation, settings);

      if (mode === 'council') {
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
            stage3_status: 'pending'
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
            chat: true
          }
        };

        updateTargetConversation((prev) => ({
          ...prev,
          messages: [...prev.messages, assistantMessage],
        }));
      }

      await api.sendMessageStream(conversationId, content, (eventType, event) => {
        switch (eventType) {
          case 'stage1_start':
            updateTargetConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.stage1 = true;
              return { ...prev, messages };
            });
            break;

          case 'stage1_complete':
            updateTargetConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.stage1 = mergeReasoningBuffersIntoResults(
                event.data,
                lastMsg.reasoningBuffers?.stage1,
              );
              lastMsg.loading.stage1 = false;

              const cost = calculateStage1Cost(lastMsg.stage1, availableModels);
              lastMsg.running_cost = (lastMsg.running_cost || 0) + cost;

              return { ...prev, messages };
            });
            break;

          case 'stage2_start':
            updateTargetConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.stage2 = true;
              return { ...prev, messages };
            });
            break;

          case 'stage2_complete':
            updateTargetConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.stage2 = mergeReasoningBuffersIntoResults(
                event.data,
                lastMsg.reasoningBuffers?.stage2,
              );
              lastMsg.metadata = event.metadata;
              lastMsg.loading.stage2 = false;

              const cost = calculateStage2Cost(lastMsg.stage2, availableModels);
              lastMsg.running_cost = (lastMsg.running_cost || 0) + cost;

              return { ...prev, messages };
            });
            break;

          case 'stage3_start':
            updateTargetConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.stage3 = true;
              return { ...prev, messages };
            });
            break;

          case 'stage3_complete':
            updateTargetConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.stage3 = mergeReasoningBufferIntoResult(
                event.data,
                lastMsg.reasoningBuffers?.stage3,
              );
              lastMsg.loading.stage3 = false;

              const cost = calculateStage3Cost(lastMsg.stage3, availableModels);
              lastMsg.running_cost = (lastMsg.running_cost || 0) + cost;

              return { ...prev, messages };
            });
            break;

          case 'chat_start':
            updateTargetConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.chat = true;
              return { ...prev, messages };
            });
            break;

          case 'chat_response':
            updateTargetConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              if (typeof event.data === 'string') {
                lastMsg.content = event.data;
              } else {
                lastMsg.content = event.data.content;
                if (event.data.reasoning) {
                  lastMsg.reasoning = event.data.reasoning;
                }
              }
              lastMsg.loading.chat = false;
              return { ...prev, messages };
            });
            break;

          case 'reasoning_delta':
            updateTargetConversation((prev) => {
              const messages = [...prev.messages];
              const lastIndex = messages.length - 1;
              if (lastIndex < 0) return prev;
              messages[lastIndex] = appendReasoningDeltaToMessage(messages[lastIndex], event.data);
              return { ...prev, messages };
            });
            break;

          case 'content_delta':
            updateTargetConversation((prev) => {
              const messages = [...prev.messages];
              const lastIndex = messages.length - 1;
              if (lastIndex < 0) return prev;
              messages[lastIndex] = appendContentDeltaToMessage(messages[lastIndex], event.data);
              return { ...prev, messages };
            });
            break;

          case 'title_complete':
            loadConversations();
            break;

          case 'budget_warning':
            setBudgetWarning(event.data);
            break;

          case 'complete':
            if (event.data) {
              updateTargetConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];
                if (lastMsg?.role === 'assistant' && event.data.turn_cost != null) {
                  lastMsg.running_cost = event.data.turn_cost;
                }

                return {
                  ...prev,
                  messages,
                  total_cost: event.data.total_cost,
                  session_usage: event.data.session_usage,
                  budget_spent_pct: event.data.budget_spent_pct,
                };
              });
            }
            loadConversations();
            setIsLoading(false);
            break;

          case 'error':
            console.error('Stream error:', event.message);
            updateTargetConversation(markLastAssistantStreamInterrupted);
            setIsLoading(false);
            break;

          default:
            console.warn('Unknown event type:', eventType);
        }
      }, mode, attachmentIds, {
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

  const sidebarProps = {
    conversations,
    currentConversationId: conversationId,
    onSelectConversation: handleSelectConversation,
    onNewConversation: handleNewConversation,
    onShowAnalytics,
    folders,
    onCreateFolder,
    onRenameFolder,
    onDeleteFolder,
    onRenameConversation,
    onDeleteConversation: handleDeleteConversationFromSidebar,
    onMoveConversation,
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background text-foreground">
      {/* Mobile Navigation Drawer */}
      <Sheet open={isMobileMenuOpen} onOpenChange={setIsMobileMenuOpen}>
        <SheetTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden absolute top-3 left-3 z-20"
            aria-label="Open menu"
          >
            <Menu className="h-5 w-5" />
          </Button>
        </SheetTrigger>
        <SheetContent side="left" className="w-[300px] p-0">
          <SidebarContent
            {...sidebarProps}
            onItemClick={() => setIsMobileMenuOpen(false)}
          />
        </SheetContent>
      </Sheet>

      {/* Desktop Sidebar */}
      <Sidebar {...sidebarProps} />

      <main className="flex-1 flex flex-col h-full overflow-hidden relative">
        <ChatInterface
          conversation={currentConversation}
          onSendMessage={handleSendMessage}
          onUpdateSessionPolicy={handleUpdateSessionPolicy}
          onUpdateConversationPrivacy={handleUpdateConversationPrivacy}
          onUpdateThinkingEffort={handleUpdateThinkingEffort}
          budgetWarning={budgetWarning}
          onDismissBudgetWarning={() => setBudgetWarning(null)}
          isLoading={isLoading}
        />
      </main>

      <ModelSelector
        isOpen={isModelSelectorOpen}
        onClose={() => setIsModelSelectorOpen(false)}
        onConfirm={handleModelConfirm}
        defaultBudgetUsd={settings.defaultSessionBudgetUsd}
      />
      {showAnalytics && (
        <AnalyticsDashboard onClose={onCloseAnalytics} />
      )}
      <Toaster />
    </div>
  );
}

function AppContent() {
  const [conversations, setConversations] = useState([]);
  const [folders, setFolders] = useState([]);
  const [availableModels, setAvailableModels] = useState([]);
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [configStatus, setConfigStatus] = useState({ loading: true, hasApiKey: null, error: null });
  const [showFirstRunSetup, setShowFirstRunSetup] = useState(false);
  const [openModelPickerSignal, setOpenModelPickerSignal] = useState(0);
  const [lastConsumedModelPickerSignal, setLastConsumedModelPickerSignal] = useState(0);
  const [configStatusRetryTick, setConfigStatusRetryTick] = useState(0);
  const configStatusRetryAttempt = useRef(0);
  const hasShownConfigStatusError = useRef(false);
  const location = useLocation();
  const { updateSettings } = useSettings();
  const isAppRoute = location.pathname === '/app' || location.pathname.startsWith('/c/');

  // Load models on mount for pricing
  useEffect(() => {
    api.getModels().then(data => setAvailableModels(data.models)).catch(console.error);
  }, []);

  useEffect(() => {
    if (!isAppRoute) return undefined;

    let cancelled = false;
    let retryTimer = null;

    api.getConfigStatus()
      .then((status) => {
        if (cancelled) return;
        const nextState = buildConfigStatusSuccessState(status);
        setConfigStatus(nextState.configStatus);
        setShowFirstRunSetup(nextState.showFirstRunSetup);
        configStatusRetryAttempt.current = 0;
        hasShownConfigStatusError.current = false;
      })
      .catch((error) => {
        if (cancelled) return;
        console.error('Failed to check configuration status:', error);
        const nextState = buildConfigStatusFailureState();
        setConfigStatus(nextState.configStatus);
        setShowFirstRunSetup(nextState.showFirstRunSetup);
        if (!hasShownConfigStatusError.current) {
          toast({
            title: 'Configuration status unavailable',
            description: 'The app will keep running and retry automatically.',
          });
          hasShownConfigStatusError.current = true;
        }
        const retryDelayMs = getConfigStatusRetryDelayMs(configStatusRetryAttempt.current);
        configStatusRetryAttempt.current += 1;
        retryTimer = window.setTimeout(() => {
          setConfigStatusRetryTick((tick) => tick + 1);
        }, retryDelayMs);
      });

    return () => {
      cancelled = true;
      if (retryTimer) {
        window.clearTimeout(retryTimer);
      }
    };
  }, [isAppRoute, configStatusRetryTick]);

  // Load conversations and folders on mount
  useEffect(() => {
    const loadData = async () => {
      try {
        const [convs, flds] = await Promise.all([
          api.listConversations(),
          api.listFolders().catch(() => []),
        ]);
        setConversations(convs);
        setFolders(flds);
      } catch (error) {
        console.error('Failed to load data:', error);
      }
    };
    loadData();
  }, []);

  // ─── Folder handlers ───────────────────────────────────────
  const handleCreateFolder = async (name) => {
    try {
      const folder = await api.createFolder(name);
      setFolders((prev) => [...prev, folder]);
    } catch (e) { console.error('Failed to create folder:', e); }
  };
  const handleRenameFolder = async (folderId, newName) => {
    try {
      await api.updateFolder(folderId, { name: newName });
      setFolders((prev) => prev.map((f) => f.id === folderId ? { ...f, name: newName } : f));
    } catch (e) { console.error('Failed to rename folder:', e); }
  };
  const handleDeleteFolder = async (folderId) => {
    try {
      await api.deleteFolder(folderId);
      setFolders((prev) => prev.filter((f) => f.id !== folderId));
      // Unset folder_id on orphaned conversations
      setConversations((prev) => prev.map((c) => c.folder_id === folderId ? { ...c, folder_id: null } : c));
    } catch (e) { console.error('Failed to delete folder:', e); }
  };

  // ─── Conversation management handlers ──────────────────────
  const handleRenameConversation = async (convId, newTitle) => {
    try {
      await api.updateConversation(convId, { title: newTitle });
      setConversations((prev) => prev.map((c) => c.id === convId ? { ...c, title: newTitle } : c));
    } catch (e) { console.error('Failed to rename conversation:', e); }
  };
  const handleDeleteConversation = async (convId) => {
    try {
      await api.deleteConversation(convId);
      setConversations((prev) => prev.filter((c) => c.id !== convId));
    } catch (e) { console.error('Failed to delete conversation:', e); }
  };
  const handleMoveConversation = async (convId, folderId) => {
    try {
      await api.updateConversation(convId, { folder_id: folderId });
      setConversations((prev) => prev.map((c) => c.id === convId ? { ...c, folder_id: folderId } : c));
    } catch (e) { console.error('Failed to move conversation:', e); }
  };

  const handleFirstRunComplete = (settingsUpdate) => {
    updateSettings(settingsUpdate);
    setConfigStatus({ loading: false, hasApiKey: true });
    setShowFirstRunSetup(false);
    setOpenModelPickerSignal((value) => value + 1);
  };

  return (
    <>
      <Routes>
        <Route
          path="/"
          element={<LandingPage />}
        />
        <Route
          path="/app"
          element={
            <ConversationView
              conversations={conversations}
              onConversationsChange={setConversations}
              availableModels={availableModels}
              onShowAnalytics={() => setShowAnalytics(true)}
              showAnalytics={showAnalytics}
              onCloseAnalytics={() => setShowAnalytics(false)}
              folders={folders}
              onCreateFolder={handleCreateFolder}
              onRenameFolder={handleRenameFolder}
              onDeleteFolder={handleDeleteFolder}
              onRenameConversation={handleRenameConversation}
              onDeleteConversation={handleDeleteConversation}
              onMoveConversation={handleMoveConversation}
              openModelPickerSignal={openModelPickerSignal}
              lastConsumedModelPickerSignal={lastConsumedModelPickerSignal}
              onConsumeModelPickerSignal={setLastConsumedModelPickerSignal}
            />
          }
        />
        <Route
          path="/c/:conversationId"
          element={
            <ConversationView
              conversations={conversations}
              onConversationsChange={setConversations}
              availableModels={availableModels}
              onShowAnalytics={() => setShowAnalytics(true)}
              showAnalytics={showAnalytics}
              onCloseAnalytics={() => setShowAnalytics(false)}
              folders={folders}
              onCreateFolder={handleCreateFolder}
              onRenameFolder={handleRenameFolder}
              onDeleteFolder={handleDeleteFolder}
              onRenameConversation={handleRenameConversation}
              onDeleteConversation={handleDeleteConversation}
              onMoveConversation={handleMoveConversation}
              openModelPickerSignal={openModelPickerSignal}
              lastConsumedModelPickerSignal={lastConsumedModelPickerSignal}
              onConsumeModelPickerSignal={setLastConsumedModelPickerSignal}
            />
          }
        />
      </Routes>
      {isAppRoute && !configStatus.loading && showFirstRunSetup && (
        <FirstRunSetup isOpen={showFirstRunSetup} onComplete={handleFirstRunComplete} />
      )}
    </>
  );
}

function App() {
  return (
    <SettingsProvider>
      <AppContent />
    </SettingsProvider>
  );
}

export default App;
