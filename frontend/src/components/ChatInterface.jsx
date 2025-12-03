import { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import Stage1 from './Stage1';
import Stage2 from './Stage2';
import Stage3 from './Stage3';
import { api } from '../api';
import './ChatInterface.css';

export default function ChatInterface({
  conversation,
  onSendMessage,
  isLoading,
}) {
  const [input, setInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [files, setFiles] = useState([]);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);
  const messagesEndRef = useRef(null);

  const [models, setModels] = useState([]);
  const [estimatedCost, setEstimatedCost] = useState(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
    calculateCost();
  }, [conversation, input, models]);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const data = await api.getModels();
      setModels(data.models);
    } catch (error) {
      console.error('Failed to load models for pricing:', error);
    }
  };

  const calculateCost = () => {
    if (!conversation || !models.length) return;

    const charCount = input.length;
    const estimatedTokens = charCount / 4; // Rough estimate

    // Determine active models based on mode
    // If no messages, it's Council Mode (Stage 1 + Stage 2 + Stage 3)
    // Actually, Stage 1 queries Council. Stage 2 queries Council. Stage 3 queries Chairman.
    // So Council models are queried TWICE (Stage 1 & 2), Chairman ONCE (Stage 3).
    // Wait, Stage 2 input is the *responses* from Stage 1, not the user input.
    // But for the *User Input* cost, we only count the input tokens sent to them.
    // Stage 1: User Input -> Council Models
    // Stage 2: User Input (in prompt) -> Council Models
    // Stage 3: User Input (in prompt) -> Chairman Model

    // So for the USER INPUT part of the cost:
    // Council Mode: Input * (Council_Count * 2 + Chairman_Count)
    // Chat Mode: Input * Chairman_Count

    const metadata = conversation.metadata || {};
    const councilIds = metadata.council_models || []; // Default might be empty if old conv
    const chairmanId = metadata.chairman_model;

    // If metadata is missing (legacy), we might not be able to estimate accurately.
    // We'll skip or assume defaults if we really wanted to, but skipping is safer.
    if (!councilIds.length && !chairmanId) {
      setEstimatedCost(null);
      return;
    }

    let totalInputRate = 0;

    const isFollowUp = conversation.messages.length > 0;

    if (!isFollowUp) {
      // Council Mode
      // 1. Council Models (Stage 1)
      councilIds.forEach(id => {
        const m = models.find(mod => mod.id === id);
        if (m) totalInputRate += m.pricing.input;
      });

      // 2. Council Models (Stage 2 - Ranking)
      // They see the user query again in the prompt
      councilIds.forEach(id => {
        const m = models.find(mod => mod.id === id);
        if (m) totalInputRate += m.pricing.input;
      });

      // 3. Chairman Model (Stage 3 - Synthesis)
      const chair = models.find(mod => mod.id === chairmanId);
      if (chair) totalInputRate += chair.pricing.input;

    } else {
      // Chat Mode (Follow-up)
      // Only Chairman sees the new input
      const chair = models.find(mod => mod.id === chairmanId);
      if (chair) totalInputRate += chair.pricing.input;
    }

    // Cost = (Tokens / 1M) * Rate
    const cost = (estimatedTokens / 1000000) * totalInputRate;
    setEstimatedCost(cost);
  };

  // Old handlers removed to avoid duplication


  const handleFileUpload = async (e) => {
    const selectedFiles = Array.from(e.target.files || []);
    if (selectedFiles.length === 0) return;

    const MAX_FILE_SIZE = 25 * 1024 * 1024; // 25MB - match backend
    const MAX_FILES = 10;

    // Validate file sizes
    const oversized = selectedFiles.filter(f => f.size > MAX_FILE_SIZE);
    if (oversized.length > 0) {
      alert(`Some files exceed ${MAX_FILE_SIZE / 1024 / 1024}MB limit: ${oversized.map(f => f.name).join(', ')}`);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      return;
    }

    // Validate total file count
    if (files.length + selectedFiles.length > MAX_FILES) {
      alert(`Maximum ${MAX_FILES} files allowed`);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      return;
    }

    // Add to state instead of immediate upload
    setFiles(prev => [...prev, ...selectedFiles]);

    // Clear input to allow selecting same file again
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeFile = (indexToRemove) => {
    setFiles(prev => prev.filter((_, index) => index !== indexToRemove));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if ((!input.trim() && files.length === 0) || isLoading || isUploading) return;

    let messageContent = input;
    let uploadedFilesContent = [];

    // Process files if any
    if (files.length > 0) {
      setIsUploading(true);
      try {
        // Upload all files sequentially (or parallel if backend supports)
        for (const file of files) {
          const result = await api.uploadFile(file);

          if (result.error) {
            throw new Error(`Failed to process ${file.name}: ${result.error}`);
          }

          const docBlock = `[Attached file: ${result.filename}]\n--- DOCUMENT START ---\n${result.text}\n--- DOCUMENT END ---${result.truncated ? '\n(Content truncated to first 50k characters)' : ''}`;
          uploadedFilesContent.push(docBlock);
        }
      } catch (error) {
        alert(error.message);
        setIsUploading(false);
        return; // Block sending on error
      }
      setIsUploading(false);
    }

    // Append file content to message
    if (uploadedFilesContent.length > 0) {
      const delimiter = "\n\n";
      messageContent = messageContent ? `${messageContent}${delimiter}${uploadedFilesContent.join(delimiter)}` : uploadedFilesContent.join(delimiter);
    }

    onSendMessage(messageContent);
    setInput('');
    setFiles([]);

    // Reset height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleInput = (e) => {
    setInput(e.target.value);
    // Auto-resize
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
  };

  const handleKeyDown = (e) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleExport = () => {
    if (!conversation) return;

    const { title, messages, created_at } = conversation;
    let md = `# ${title}\nDate: ${new Date(created_at).toLocaleString()}\n\n`;

    messages.forEach(msg => {
      md += `## ${msg.role === 'user' ? 'User' : 'LLM Council'}\n\n`;

      if (msg.role === 'user') {
        md += `${msg.content}\n\n`;
      } else {
        if (msg.stage3) {
          md += `### Final Answer\n${msg.stage3.response}\n\n`;
          md += `#### Stage 1 Responses\n`;
          msg.stage1.forEach(r => md += `- **${r.model}**: ${r.response.substring(0, 100)}...\n`);
        } else {
          md += `${msg.content}\n\n`;
        }
      }
      md += `---\n\n`;
    });

    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (!conversation) {
    return (
      <div className="chat-interface">
        <div className="empty-state">
          <h2>Welcome to LLM Council</h2>
          <p>Create a new conversation to get started</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h3>{conversation.title}</h3>
        <button className="export-button" onClick={handleExport} title="Export to Markdown">
          📥 Export
        </button>
      </div>
      <div className="messages-container">
        {conversation.messages.length === 0 ? (
          <div className="empty-state">
            <h2>Start a conversation</h2>
            <p>Ask a question to consult the LLM Council</p>
          </div>
        ) : (
          conversation.messages.map((msg, index) => (
            <div key={index} className="message-group">
              {msg.role === 'user' ? (
                <div className="user-message">
                  <div className="message-label">You</div>
                  <div className="message-content">
                    <div className="markdown-content">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="assistant-message">
                  <div className="message-label">LLM Council</div>

                  {/* Stage 1 */}
                  {msg.loading?.stage1 && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>Running Stage 1: Collecting individual responses...</span>
                    </div>
                  )}
                  {msg.stage1 && <Stage1 responses={msg.stage1} />}

                  {/* Stage 2 */}
                  {msg.loading?.stage2 && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>Running Stage 2: Peer rankings...</span>
                    </div>
                  )}
                  {msg.stage2 && (
                    <Stage2
                      rankings={msg.stage2}
                      labelToModel={msg.metadata?.label_to_model}
                      aggregateRankings={msg.metadata?.aggregate_rankings}
                    />
                  )}

                  {/* Stage 3 */}
                  {msg.loading?.stage3 && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>Running Stage 3: Final synthesis...</span>
                    </div>
                  )}
                  {msg.stage3 && <Stage3 finalResponse={msg.stage3} />}

                  {/* Simple Chat Content (Follow-up) */}
                  {msg.loading?.chat && !msg.content && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>Chairman is thinking...</span>
                    </div>
                  )}
                  {msg.reasoning && (
                    <details className="reasoning-details">
                      <summary>💭 Chain of Thought (Click to expand)</summary>
                      <div className="reasoning-content">
                        <ReactMarkdown>{msg.reasoning}</ReactMarkdown>
                      </div>
                    </details>
                  )}
                  {msg.content && (
                    <div className="message-content">
                      <div className="markdown-content">
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))
        )}

        {isLoading && (
          <div className="loading-indicator">
            <div className="spinner"></div>
            <span>Processing...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form className="input-form" onSubmit={handleSubmit}>
        {/* File List */}
        {files.length > 0 && (
          <div className="file-list">
            {files.map((file, index) => (
              <div key={index} className="file-item">
                <span className="file-name">{file.name}</span>
                <button
                  type="button"
                  className="remove-file"
                  onClick={() => removeFile(index)}
                  title="Remove file"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="input-container">
          <div className="input-top-row">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              style={{ display: 'none' }}
              accept=".pdf,.docx,.txt,.md,image/*"
              multiple
            />
            <button
              type="button"
              className="attach-button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading || isUploading}
              title="Attach files"
            >
              📎
            </button>
            <textarea
              ref={textareaRef}
              className="message-input auto-resize"
              placeholder="Ask your question... (Shift+Enter for new line, Enter to send)"
              value={input}
              onChange={handleInput}
              onKeyDown={handleKeyDown}
              disabled={isLoading || isUploading}
              rows={1}
            />
            <button
              type="submit"
              className="send-button"
              disabled={(!input.trim() && files.length === 0) || isLoading || isUploading}
            >
              {isUploading ? '⏳' : 'Send'}
            </button>
          </div>
          <div className="input-bottom-row">
            <div className="budget-estimate">
              {estimatedCost !== null && (
                <span title="Rough estimate based on input characters">
                  Est. Next: ${estimatedCost.toFixed(6)}
                </span>
              )}
              {conversation.total_cost !== undefined && (
                <span className="total-cost">
                  {' | '}Total: ${conversation.total_cost.toFixed(4)}
                </span>
              )}
            </div>
          </div>
        </div>
      </form>
    </div>
  );
}
