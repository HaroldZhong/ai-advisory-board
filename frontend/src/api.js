/**
 * API client for the AI Advisory Board backend.
 */

// Use environment variable for API URL, fallback to localhost for local development
// In production with reverse proxy or packaged exe, set API_BASE to empty string
const API_BASE = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://localhost:8001' : '');

async function buildApiError(response, fallbackMessage) {
  const errorData = await response.json().catch(() => ({}));
  const error = new Error(errorData.detail || fallbackMessage);
  error.status = response.status;
  return error;
}

export const api = {
  /**
   * List all conversations.
   */
  async listConversations() {
    const response = await fetch(`${API_BASE}/api/conversations`);
    if (!response.ok) {
      throw new Error('Failed to list conversations');
    }
    return response.json();
  },

  /**
   * Create a new conversation.
   */
  async createConversation(topic, councilMembers = null, chairmanModel = null, options = {}) {
    const response = await fetch(`${API_BASE}/api/conversations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        topic,
        council_members: councilMembers,
        chairman_model: chairmanModel,
        preset_id: options.presetId || null,
        zdr_enabled: options.zdrEnabled ?? null,
        budget_usd: options.budgetUsd ?? null,
        // v1.3.0 D3: user-owned budget -- new conversations allow overage by
        // default (warn, don't block); the hard cap is opt-in.
        budget_allow_overage: options.budgetAllowOverage ?? true,
        default_mode: options.defaultMode ?? null,
      }),
    });
    if (!response.ok) {
      throw new Error('Failed to create conversation');
    }
    return response.json();
  },

  /**
   * Get a specific conversation.
   */
  async getConversation(conversationId) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}`
    );
    if (!response.ok) {
      throw new Error('Failed to get conversation');
    }
    return response.json();
  },

  /**
   * Export a conversation to Markdown on the backend and return the saved path.
   */
  async exportConversation(conversationId) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/export`,
      { method: 'POST' }
    );
    if (!response.ok) {
      throw await buildApiError(response, 'Failed to export conversation');
    }
    return response.json();
  },

  /**
   * Get session budget policy and usage for a conversation.
   */
  async getSessionPolicy(conversationId) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/session-policy`
    );
    if (!response.ok) {
      throw new Error('Failed to get session policy');
    }
    return response.json();
  },

  /**
   * v1.3.0 D3 soft seatbelt: an APPROXIMATE pre-send cost estimate for the next
   * turn. Returns { predicted_cost, approximate, threshold, is_large }. POSTs the
   * pending message + routing so the backend runs the SAME task-signal routing as
   * the send path (a long/file/research turn on auto mode is not under-estimated).
   */
  async getTurnEstimate(conversationId, { content = '', hasAttachments = false, mode = 'council', executionMode, ragPreset, modelTier } = {}) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/estimate`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content,
          has_attachments: hasAttachments,
          mode,
          execution_mode: executionMode || 'auto',
          rag_preset: ragPreset || 'auto',
          model_tier: modelTier || 'auto',
        }),
      }
    );
    if (!response.ok) {
      throw new Error('Failed to get turn estimate');
    }
    return response.json();
  },

  /**
   * Update session budget policy for a conversation.
   */
  async updateSessionPolicy(conversationId, policy) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/session-policy`,
      {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(policy),
      }
    );
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Failed to update session policy');
    }
    return response.json();
  },

  /**
   * Send a message in a conversation.
   */
  async sendMessage(conversationId, content, mode = 'auto', options = {}) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/message`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content,
          mode,
          zdr_enabled: options.zdrEnabled || false,
          execution_mode: options.executionMode || 'auto',
          rag_preset: options.ragPreset || 'auto',
          model_tier: options.modelTier || 'auto',
        }),
      }
    );
    if (!response.ok) {
      throw await buildApiError(response, 'Failed to send message');
    }
    return response.json();
  },

  /**
   * Send a message and receive streaming updates.
   * @param {string} conversationId - The conversation ID
   * @param {string} content - The message content
   * @param {function} onEvent - Callback function for each event: (eventType, data) => void
   * @param {string} mode - The mode: 'auto', 'council', or 'chat'
   * @param {string[]} attachmentIds - Optional list of attachment IDs to include
   * @returns {Promise<void>}
   */
  async sendMessageStream(conversationId, content, onEvent, mode = 'auto', attachmentIds = [], webSearch = {}, editIndex = -1) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/message/stream`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content,
          mode,
          attachment_ids: attachmentIds,
          web_search_enabled: webSearch.enabled || false,
          web_search_depth: webSearch.depth || 'fast',
          custom_instructions: webSearch.customInstructions || '',
          zdr_enabled: webSearch.zdrEnabled || false,
          execution_mode: webSearch.executionMode || 'auto',
          rag_preset: webSearch.ragPreset || 'auto',
          model_tier: webSearch.modelTier || 'auto',
          edit_index: editIndex,
        }),
      }
    );

    if (!response.ok) {
      throw await buildApiError(response, 'Failed to send message');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = ''; // Buffer to handle partial lines across chunks

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // Append decoded chunk to buffer, using stream mode to handle multi-byte chars
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');

      // Keep the last (potentially incomplete) line in the buffer
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          try {
            const event = JSON.parse(data);
            onEvent(event.type, event);
          } catch (e) {
            console.error('Failed to parse SSE event:', e);
          }
        }
      }
    }

    // Process any remaining data in buffer after stream ends
    if (buffer.startsWith('data: ')) {
      try {
        const event = JSON.parse(buffer.slice(6));
        onEvent(event.type, event);
      } catch (e) {
        // Ignore incomplete final chunk
      }
    }
  },

  // Get available models
  async getModels() {
    const response = await fetch(`${API_BASE}/api/models`);
    if (!response.ok) {
      throw new Error('Failed to fetch models');
    }
    return response.json();
  },

  /**
   * Check if API key is configured
   */
  async getConfigStatus() {
    const response = await fetch(`${API_BASE}/api/config/status`);
    if (!response.ok) {
      throw new Error('Failed to check configuration');
    }
    return response.json();
  },

  /**
   * Check connectivity to OpenRouter and validity of a key.
   * Pass apiKey to validate a not-yet-saved key (first-run "Test connection");
   * omit it to check the currently configured key.
   * @returns {Promise<{reachable: boolean, key_valid: boolean|null, error_kind: string|null, detail: string}>}
   */
  async getConnectivity(apiKey) {
    const response = apiKey
      ? await fetch(`${API_BASE}/api/config/connectivity`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ api_key: apiKey }),
        })
      : await fetch(`${API_BASE}/api/config/connectivity`);
    if (!response.ok) {
      throw new Error('Failed to check connectivity');
    }
    return response.json();
  },

  /**
   * Save OpenRouter API Key
   */
  async setupConfig(apiKey) {
    const response = await fetch(`${API_BASE}/api/config/setup`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ api_key: apiKey }),
    });
    if (!response.ok) {
      throw new Error('Failed to save configuration');
    }
    return response.json();
  },

  /**
   * Get analytics data.
   */
  async getAnalytics() {
    const response = await fetch(`${API_BASE}/api/analytics`);
    if (!response.ok) {
      throw new Error('Failed to fetch analytics');
    }
    return response.json();
  },

  // ==========================================================================
  // ATTACHMENT API (New unified file upload system)
  // ==========================================================================

  /**
   * Upload a file and create an attachment.
   * Returns attachment metadata with status.
   * @param {File} file - The file to upload
   * @returns {Promise<{attachment_id, status, filename, cached, method, warning, error, stats}>}
   */
  async uploadAttachment(file, useZdr = false) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/api/attachments?use_zdr=${useZdr}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Failed to upload attachment');
    }

    return response.json();
  },

  /**
   * Get attachment metadata.
   * @param {string} attachmentId - The attachment ID
   * @returns {Promise<Attachment>}
   */
  async getAttachment(attachmentId) {
    const response = await fetch(`${API_BASE}/api/attachments/${attachmentId}`);
    if (!response.ok) {
      throw new Error('Failed to get attachment');
    }
    return response.json();
  },

  /**
   * Delete an attachment if it is no longer referenced by a conversation.
   * @param {string} attachmentId - The attachment ID
   * @returns {Promise<{attachment_id, deleted, retained, files_deleted}>}
   */
  async deleteAttachment(attachmentId) {
    const response = await fetch(`${API_BASE}/api/attachments/${attachmentId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Failed to delete attachment');
    }
    return response.json();
  },

  /**
   * Get attachment extracted text.
   * @param {string} attachmentId - The attachment ID
   * @param {boolean} preview - If true, returns first 1000 chars only
   * @returns {Promise<{text: string, preview: boolean}>}
   */
  async getAttachmentText(attachmentId, preview = false) {
    const url = `${API_BASE}/api/attachments/${attachmentId}/text${preview ? '?preview=true' : ''}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Failed to get attachment text');
    }
    return response.json();
  },

  /**
   * Get extraction recommendation for an attachment.
   * @param {string} attachmentId - The attachment ID
   * @returns {Promise<{needs_enhanced, recommended_engine, reason, estimated_cost, cost_hint}>}
   */
  async getExtractionRecommendation(attachmentId) {
    const response = await fetch(`${API_BASE}/api/attachments/${attachmentId}/recommendation`);
    if (!response.ok) {
      throw new Error('Failed to get recommendation');
    }
    return response.json();
  },

  /**
   * Retry extraction with OpenRouter enhanced processing.
   * @param {string} attachmentId - The attachment ID
   * @param {string} engine - "pdf-text" (free) or "mistral-ocr" (paid)
   * @param {boolean} useZdr - Enable Zero Data Retention
   * @returns {Promise<{attachment_id, status, method, char_count, cost, error}>}
   */
  async enhanceAttachment(attachmentId, engine = 'pdf-text', useZdr = false) {
    const response = await fetch(
      `${API_BASE}/api/attachments/${attachmentId}/enhance?engine=${engine}&use_zdr=${useZdr}`,
      { method: 'POST' }
    );
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Failed to enhance attachment');
    }
    return response.json();
  },

  // ==========================================================================
  // FOLDER API
  // ==========================================================================

  async listFolders() {
    const response = await fetch(`${API_BASE}/api/folders`);
    if (!response.ok) throw new Error('Failed to list folders');
    return response.json();
  },

  async createFolder(name, color = null) {
    const response = await fetch(`${API_BASE}/api/folders`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, color }),
    });
    if (!response.ok) throw new Error('Failed to create folder');
    return response.json();
  },

  async updateFolder(folderId, updates) {
    const response = await fetch(`${API_BASE}/api/folders/${folderId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    if (!response.ok) throw new Error('Failed to update folder');
    return response.json();
  },

  async deleteFolder(folderId) {
    const response = await fetch(`${API_BASE}/api/folders/${folderId}`, { method: 'DELETE' });
    if (!response.ok) throw new Error('Failed to delete folder');
    return response.json();
  },

  // ==========================================================================
  // CONVERSATION MANAGEMENT API
  // ==========================================================================

  async updateConversation(conversationId, updates) {
    const response = await fetch(`${API_BASE}/api/conversations/${conversationId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Failed to update conversation');
    }
    return response.json();
  },

  async deleteConversation(conversationId) {
    const response = await fetch(`${API_BASE}/api/conversations/${conversationId}`, { method: 'DELETE' });
    if (!response.ok) throw new Error('Failed to delete conversation');
    return response.json();
  },
};
