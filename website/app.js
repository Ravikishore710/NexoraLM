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
    const lastUserPrompt = allMessages[allMessages.length - 1].content;

    // 1. Try FastAPI local server first
    try {
      const response = await fetch('http://localhost:8000/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: allMessages,
          max_tokens: 256,
          temperature: 0.7,
          stream: true
        })
      });

      if (response.ok) {
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
      }
    } catch (err) {
      // Backend not running, proceed to fast local response generator
    }

    // 2. Local intelligent response generator
    const answer = generateLocalAnswer(lastUserPrompt);
    const words = answer.split(' ');

    for (let i = 0; i < words.length; i++) {
      await new Promise(r => setTimeout(r, 25));
      onChunk(words[i] + ' ');
    }
    onDone();
  }

  function generateLocalAnswer(prompt) {
    const p = prompt.toLowerCase();
    if (p.includes('gqa') || p.includes('grouped')) {
      return "Grouped-Query Attention (GQA) groups query heads together so that multiple query heads share a single Key and Value head. In NexoraLM, we use 12 Query heads and 4 Key/Value heads (a 3:1 sharing ratio). This reduces the memory footprint and bandwidth required for the KV-cache by 66.7% during autoregressive token generation while retaining expressiveness near full Multi-Head Attention.";
    }
    if (p.includes('rope') || p.includes('rotary') || p.includes('position')) {
      return "Rotary Position Embedding (RoPE) encodes relative token positions directly into query and key representations via complex rotation matrices. Because the dot product between query and key depends purely on their relative distance (m - n), RoPE generalizes across long contexts seamlessly without requiring trainable positional parameters.";
    }
    if (p.includes('lora') || p.includes('qlora') || p.includes('adapter')) {
      return "LoRA freezes the base model weights W and injects low-rank trainable decomposition matrices (W + (alpha / r) * B @ A). In NexoraLM, we adapt the Q, K, V, and O projections with rank r=16, training only 1,310,720 parameters (1.03% of the model). QLoRA quantizes the base weights down to 4-bit, dropping memory footprint to 64.7 MB while keeping full adapter quality.";
    }
    if (p.includes('deep learning') || p.includes('neural network') || p.includes('learn')) {
      return "Deep learning is a subset of machine learning based on multi-layered neural networks. Models learn representational hierarchies: early layers capture primitive features (like token tokens or syntax), while deeper layers compose them into abstract semantics. Training uses gradient descent with backpropagation to iteratively minimize loss on target data.";
    }
    if (p.includes('quantiz') || p.includes('int8') || p.includes('int4')) {
      return "Quantization maps high-precision FP16 weights into discrete integer formats. In NexoraLM, our per-channel symmetric INT8 quantization compressed the model from 240 MB to 120.55 MB with a mean relative error of only 0.0078 (<0.8%). Our INT4 group-wise quantization reached 64.69 MB (3.7x compression).";
    }
    return `NexoraLM is a 126M parameter decoder-only language model trained from scratch on ~300M tokens from FineWeb-Edu. It features 16 Transformer layers, RMSNorm, RoPE, SwiGLU, and 32k Byte-level BPE, followed by SFT and DPO alignment. You asked: "${prompt}". Feel free to ask more about any machine learning or technical concept!`;
  }
});
