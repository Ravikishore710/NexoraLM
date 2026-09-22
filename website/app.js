document.addEventListener('DOMContentLoaded', () => {
  const chatForm = document.getElementById('chatForm');
  const chatInput = document.getElementById('chatInput');
  const btnSend = document.getElementById('btnSend');
  const messagesContainer = document.getElementById('messagesContainer');
  const messagesList = document.getElementById('messagesList');
  const welcomeScreen = document.getElementById('welcomeScreen');
  const btnNewChat = document.getElementById('btnNewChat');
  const btnClearChat = document.getElementById('btnClearChat');
  const chatHistoryList = document.getElementById('chatHistoryList');
  const btnToggleSidebar = document.getElementById('btnToggleSidebar');
  const sidebar = document.getElementById('sidebar');
  const currentChatTitle = document.getElementById('currentChatTitle');
  const suggestionCards = document.querySelectorAll('.suggestion-card');

  // State
  let conversations = JSON.parse(localStorage.getItem('nexoralm_conversations') || '[]');
  let currentId = conversations.length > 0 ? conversations[0].id : null;
  let isGenerating = false;

  const backendStatus = document.getElementById('backendStatus');
  async function checkBackendHealth() {
    try {
      const res = await fetch('http://localhost:8000/health');
      if (res.ok) {
        backendStatus.innerHTML = '<span class="status-dot online"></span><span class="status-text">NexoraLM Model Live</span>';
      } else {
        backendStatus.innerHTML = '<span class="status-dot" style="background: #f59e0b; box-shadow: 0 0 6px #f59e0b;"></span><span class="status-text">Model Initializing</span>';
      }
    } catch (e) {
      backendStatus.innerHTML = '<span class="status-dot" style="background: #ef4444; box-shadow: 0 0 6px #ef4444;"></span><span class="status-text">Backend Offline (:8000)</span>';
    }
  }
  checkBackendHealth();
  setInterval(checkBackendHealth, 5000);


  // Auto-resize textarea
  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
    btnSend.disabled = !chatInput.value.trim() || isGenerating;
  });

  // Enter to send (Shift+Enter for newline)
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!btnSend.disabled) {
        chatForm.dispatchEvent(new Event('submit'));
      }
    }
  });

  // Toggle Sidebar
  btnToggleSidebar.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
  });

  // New Chat
  btnNewChat.addEventListener('click', () => {
    startNewChat();
  });

  // Clear Chat
  btnClearChat.addEventListener('click', () => {
    if (!currentId) return;
    const conv = conversations.find(c => c.id === currentId);
    if (conv) {
      conv.messages = [];
      saveConversations();
      renderCurrentConversation();
    }
  });

  // Prompt suggestions
  suggestionCards.forEach(card => {
    card.addEventListener('click', () => {
      const prompt = card.getAttribute('data-prompt');
      chatInput.value = prompt;
      btnSend.disabled = false;
      chatForm.dispatchEvent(new Event('submit'));
    });
  });

  // Form submit
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (!text || isGenerating) return;

    chatInput.value = '';
    chatInput.style.height = 'auto';
    btnSend.disabled = true;

    // If no current conversation, create one
    if (!currentId) {
      const newConv = {
        id: 'conv_' + Date.now(),
        title: text.slice(0, 30) + (text.length > 30 ? '...' : ''),
        messages: []
      };
      conversations.unshift(newConv);
      currentId = newConv.id;
    }

    const currentConv = conversations.find(c => c.id === currentId);
    if (!currentConv) return;

    // Update title if it's the first message
    if (currentConv.messages.length === 0) {
      currentConv.title = text.slice(0, 30) + (text.length > 30 ? '...' : '');
      currentChatTitle.textContent = currentConv.title;
    }

    // Append user message
    currentConv.messages.push({ role: 'user', content: text });
    saveConversations();
    renderCurrentConversation();

    // Prepare assistant message
    isGenerating = true;
    const asstBubble = createAssistantMessageElement();
    messagesList.appendChild(asstBubble);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    const contentDiv = asstBubble.querySelector('.chat-content');
    contentDiv.classList.add('typing-cursor');

    let fullAnswer = '';

    // Stream from backend or fallback to smart built-in answering
    await streamResponse(
      currentConv.messages,
      (chunk) => {
        fullAnswer += chunk;
        contentDiv.textContent = fullAnswer;
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
      },
      () => {
        contentDiv.classList.remove('typing-cursor');
        currentConv.messages.push({ role: 'assistant', content: fullAnswer });
        saveConversations();
        isGenerating = false;
        btnSend.disabled = !chatInput.value.trim();
      }
    );
  });

  // Initial load
  if (conversations.length === 0) {
    startNewChat();
  } else {
    renderHistoryList();
    renderCurrentConversation();
  }

  // Helpers
  function startNewChat() {
    currentId = null;
    currentChatTitle.textContent = 'New Chat';
    messagesList.innerHTML = '';
    welcomeScreen.style.display = 'flex';
    renderHistoryList();
    chatInput.focus();
  }

  function saveConversations() {
    localStorage.setItem('nexoralm_conversations', JSON.stringify(conversations));
    renderHistoryList();
  }

  function renderHistoryList() {
    chatHistoryList.innerHTML = '';
    conversations.forEach(conv => {
      const item = document.createElement('div');
      item.className = 'history-item' + (conv.id === currentId ? ' active' : '');
      item.textContent = conv.title || 'Untitled Conversation';
      item.addEventListener('click', () => {
        currentId = conv.id;
        renderHistoryList();
        renderCurrentConversation();
      });
      chatHistoryList.appendChild(item);
    });
  }

  function renderCurrentConversation() {
    const conv = conversations.find(c => c.id === currentId);
    if (!conv || conv.messages.length === 0) {
      welcomeScreen.style.display = 'flex';
      messagesList.innerHTML = '';
      currentChatTitle.textContent = 'New Chat';
      return;
    }

    welcomeScreen.style.display = 'none';
    currentChatTitle.textContent = conv.title;
    messagesList.innerHTML = '';

    conv.messages.forEach(msg => {
      const row = document.createElement('div');
      row.className = `chat-bubble-row ${msg.role}`;

      if (msg.role === 'assistant') {
        const avatar = document.createElement('div');
        avatar.className = 'chat-avatar assistant';
        avatar.textContent = 'N';
        row.appendChild(avatar);
      }

      const content = document.createElement('div');
      content.className = 'chat-content';
      content.textContent = msg.content;
      row.appendChild(content);

      messagesList.appendChild(row);
    });

    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function createAssistantMessageElement() {
    const row = document.createElement('div');
    row.className = 'chat-bubble-row assistant';

    const avatar = document.createElement('div');
    avatar.className = 'chat-avatar assistant';
    avatar.textContent = 'N';
    row.appendChild(avatar);

    const content = document.createElement('div');
    content.className = 'chat-content';
    row.appendChild(content);

    return row;
  }

  async function streamResponse(allMessages, onChunk, onDone) {
    try {
      const response = await fetch('http://localhost:8000/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: allMessages,
          max_tokens: 180,
          temperature: 0.8,
          top_p: 0.92,
          top_k: 50,
          repetition_penalty: 1.25,
          stream: true
        })
      });


      if (!response.ok) {
        throw new Error(`NexoraLM Backend HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('data: ')) {
            const dataStr = trimmed.slice(6);
            if (dataStr === '[DONE]') {
              onDone();
              return;
            }
            try {
              const parsed = JSON.parse(dataStr);
              const delta = parsed.choices?.[0]?.delta?.content || parsed.choices?.[0]?.text || '';
              if (delta) onChunk(delta);
            } catch (e) {}
          }
        }
      }
      onDone();
      return;
    } catch (err) {
      console.error('Inference error:', err);
      onChunk(`[Error connecting to NexoraLM backend]: ${err.message}. Please ensure the server is active: 'python scripts/serve.py --checkpoint checkpoints/nexoralm_aligned_300m.pt --port 8000'`);
      onDone();
    }
  }
});

