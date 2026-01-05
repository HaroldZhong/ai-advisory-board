/**
 * API client for the AI Advisory Board backend.
 */

// Use environment variable for API URL, fallback to localhost for local development
// In production with reverse proxy, set VITE_API_URL to empty string or omit it
const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8001';

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
  async createConversation(topic, councilMembers = null, chairmanModel = null) {
    const response = await fetch(`${API_BASE}/api/conversations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        topic,
        council_members: councilMembers,
        chairman_model: chairmanModel
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
   * Send a message in a conversation.
   */
  async sendMessage(conversationId, content, mode = 'auto') {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/message`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content, mode }),
      }
    );
    if (!response.ok) {
      throw new Error('Failed to send message');
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
  async sendMessageStream(conversationId, content, onEvent, mode = 'auto', attachmentIds = []) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/message/stream`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content, mode, attachment_ids: attachmentIds }),
      }
    );

    if (!response.ok) {
      throw new Error('Failed to send message');
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

  /**
   * Upload a file and get extracted text.
   * @param {File} file - The file to upload
   * @returns {Promise<{text: string, filename: string, truncated: boolean}>}
   */
  async uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Failed to upload file');
    }

    return response.json();
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
  async uploadAttachment(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/api/attachments`, {
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
};
