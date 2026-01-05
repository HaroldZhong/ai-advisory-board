import { useState, useEffect } from 'react';
import { Routes, Route, useParams, useNavigate } from 'react-router-dom';
// LandingPage import removed for isolated marketing branch
import Sidebar, { SidebarContent } from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import { api } from './api';
import { Toaster } from "@/components/ui/toaster";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Menu } from "lucide-react";

import ModelSelector from './components/ModelSelector';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import { calculateUsageCost, calculateStage1Cost, calculateStage2Cost, calculateStage3Cost } from './utils/cost';
import { SettingsProvider } from './contexts/SettingsContext';

function ConversationView({
  conversations,
  onConversationsChange,
  availableModels,
  onShowAnalytics,
  showAnalytics,
  onCloseAnalytics
}) {
  const { conversationId } = useParams();
  const navigate = useNavigate();

  const [currentConversation, setCurrentConversation] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isModelSelectorOpen, setIsModelSelectorOpen] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

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

  const handleNewConversation = () => {
    setIsModelSelectorOpen(true);
  };

  const handleModelConfirm = async (councilMembers, chairmanModel) => {
    try {
      const newConv = await api.createConversation("New Conversation", councilMembers, chairmanModel);

      // Update conversations list
      onConversationsChange([
        { id: newConv.id, title: "New Conversation", created_at: newConv.created_at, message_count: 0 },
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

  const loadConversations = async () => {
    try {
      const convs = await api.listConversations();
      onConversationsChange(convs);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  };

  const handleSendMessage = async (content, attachmentIds = [], attachmentMetadata = []) => {
    if (!conversationId) return;

    setIsLoading(true);
    try {
      const userMessage = {
        role: 'user',
        content,
        attachments: attachmentMetadata
      };

      const isFollowUp = currentConversation.messages.length > 0;
      const mode = isFollowUp ? 'chat' : 'council';

      setCurrentConversation((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage],
      }));

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

        setCurrentConversation((prev) => ({
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

        setCurrentConversation((prev) => ({
          ...prev,
          messages: [...prev.messages, assistantMessage],
        }));
      }

      await api.sendMessageStream(conversationId, content, (eventType, event) => {
        switch (eventType) {
          case 'stage1_start':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.stage1 = true;
              return { ...prev, messages };
            });
            break;

          case 'stage1_complete':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.stage1 = event.data;
              lastMsg.loading.stage1 = false;

              const cost = calculateStage1Cost(event.data, availableModels);
              lastMsg.running_cost = (lastMsg.running_cost || 0) + cost;

              return { ...prev, messages };
            });
            break;

          case 'stage2_start':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.stage2 = true;
              return { ...prev, messages };
            });
            break;

          case 'stage2_complete':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.stage2 = event.data;
              lastMsg.metadata = event.metadata;
              lastMsg.loading.stage2 = false;

              const cost = calculateStage2Cost(event.data, availableModels);
              lastMsg.running_cost = (lastMsg.running_cost || 0) + cost;

              return { ...prev, messages };
            });
            break;

          case 'stage3_start':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.stage3 = true;
              return { ...prev, messages };
            });
            break;

          case 'stage3_complete':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.stage3 = event.data;
              lastMsg.loading.stage3 = false;

              const cost = calculateStage3Cost(event.data, availableModels);
              lastMsg.running_cost = (lastMsg.running_cost || 0) + cost;

              return { ...prev, messages };
            });
            break;

          case 'chat_start':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.chat = true;
              return { ...prev, messages };
            });
            break;

          case 'chat_response':
            setCurrentConversation((prev) => {
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

          case 'title_complete':
            loadConversations();
            break;

          case 'complete':
            if (event.data) {
              setCurrentConversation((prev) => ({
                ...prev,
                total_cost: event.data.total_cost,
              }));
            }
            loadConversations();
            setIsLoading(false);
            break;

          case 'error':
            console.error('Stream error:', event.message);
            setIsLoading(false);
            break;

          default:
            console.warn('Unknown event type:', eventType);
        }
      }, mode, attachmentIds);
    } catch (error) {
      console.error('Failed to send message:', error);
      alert(`Failed to send message: ${error.message || 'Unknown error'}`);
      setCurrentConversation((prev) => {
        const messages = [...prev.messages];
        if (messages.length >= 2 && messages[messages.length - 1].role === 'assistant') {
          messages.splice(-2);
        } else if (messages.length >= 1 && messages[messages.length - 1].role === 'user') {
          messages.splice(-1);
        }
        return { ...prev, messages };
      });
      setIsLoading(false);
    }
  };

  const sidebarProps = {
    conversations,
    currentConversationId: conversationId,
    onSelectConversation: handleSelectConversation,
    onNewConversation: handleNewConversation,
    onShowAnalytics,
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
          isLoading={isLoading}
        />
      </main>

      <ModelSelector
        isOpen={isModelSelectorOpen}
        onClose={() => setIsModelSelectorOpen(false)}
        onConfirm={handleModelConfirm}
      />
      {showAnalytics && (
        <AnalyticsDashboard onClose={onCloseAnalytics} />
      )}
      <Toaster />
    </div>
  );
}

function App() {
  const [conversations, setConversations] = useState([]);
  const [availableModels, setAvailableModels] = useState([]);
  const [showAnalytics, setShowAnalytics] = useState(false);

  // Load models on mount for pricing
  useEffect(() => {
    api.getModels().then(data => setAvailableModels(data.models)).catch(console.error);
  }, []);

  // Load conversations on mount
  useEffect(() => {
    const loadConversations = async () => {
      try {
        const convs = await api.listConversations();
        setConversations(convs);
      } catch (error) {
        console.error('Failed to load conversations:', error);
      }
    };
    loadConversations();
  }, []);

  return (
    <SettingsProvider>
      <Routes>
        <Route
          path="/"
          element={
            <ConversationView
              conversations={conversations}
              onConversationsChange={setConversations}
              availableModels={availableModels}
              onShowAnalytics={() => setShowAnalytics(true)}
              showAnalytics={showAnalytics}
              onCloseAnalytics={() => setShowAnalytics(false)}
            />
          }
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
            />
          }
        />
      </Routes>
    </SettingsProvider>
  );
}

export default App;
