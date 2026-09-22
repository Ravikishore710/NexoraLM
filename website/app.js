document.addEventListener('DOMContentLoaded', () => {
  const chatForm = document.getElementById('chatForm');
  const chatInput = document.getElementById('chatInput');
  const chatMessages = document.getElementById('chatMessages');
  const tokenStreamBox = document.getElementById('tokenStreamBox');
  const ttftDisplay = document.getElementById('ttft');
  const presetButtons = document.querySelectorAll('.preset-btn');

  const knowledgeBase = {
    'gqa': 'Grouped-Query Attention (GQA) shares key and value projection heads across multiple query heads. In NexoraLM, we use 12 Query heads and 4 Key/Value heads (a 3:1 sharing ratio). This reduces the memory bandwidth requirement and footprint of the KV-cache by 66.7% during autoregressive generation while maintaining expressiveness near standard Multi-Head Attention.',
    'rope': 'Rotary Position Embedding (RoPE) encodes absolute token positions into complex rotation matrices applied directly to queries and keys. Because the inner product between query and key depends strictly on their relative distance (m - n), RoPE exhibits excellent length extrapolation properties without needing trainable positional embedding parameters.',
    'lora': 'LoRA freezes the base model weights W and injects low-rank trainable decomposition matrices: W + (alpha / r) * (B @ A). In NexoraLM, we apply rank r=16 to Q, K, V, and O projections, creating only 1,310,720 trainable parameters (1.03% of the model). QLoRA goes a step further by quantizing the frozen base weights to 4-bit, drastically reducing VRAM while maintaining full FP16 adapter precision.',
    'quantization': 'Post-training quantization reduces parameter bit-width. In NexoraLM, our INT8 per-channel symmetric quantization compressed weights from 240 MB down to 120.55 MB with a mean relative error of just 0.0078 (<0.8%). Our INT4 group-wise (group size 128) achieved 64.69 MB (3.7x compression) with ~10% relative distortion.',
    'default': 'NexoraLM is a 126M parameter decoder-only language model trained on ~300M tokens from FineWeb-Edu. It features RMSNorm, RoPE, 16 layers, SwiGLU feed-forward networks, and tied input/output embeddings, followed by SFT, LoRA, and DPO alignment.'
  };

  presetButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const prompt = btn.getAttribute('data-prompt');
      chatInput.value = prompt;
      handleSubmit(prompt);
    });
  });

  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (!text) return;
    chatInput.value = '';
    handleSubmit(text);
  });

  function appendMessage(role, content) {
    const msg = document.createElement('div');
    msg.className = `msg ${role}`;
    msg.textContent = content;
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msg;
  }

  function determineResponse(prompt) {
    const lower = prompt.toLowerCase();
    if (lower.includes('gqa') || lower.includes('grouped')) return knowledgeBase['gqa'];
    if (lower.includes('rope') || lower.includes('rotary') || lower.includes('position')) return knowledgeBase['rope'];
    if (lower.includes('lora') || lower.includes('qlora') || lower.includes('adapter')) return knowledgeBase['lora'];
    if (lower.includes('quant') || lower.includes('int8') || lower.includes('int4')) return knowledgeBase['quantization'];
    return knowledgeBase['default'];
  }

  async function queryLiveApi(prompt, onChunk, onDone, onError) {
    try {
      const response = await fetch('http://localhost:8000/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{ role: 'user', content: prompt }],
          max_tokens: 256,
          temperature: 0.7,
          stream: true
        })
      });

      if (!response.ok) throw new Error(`API returned ${response.status}`);

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
              return true;
            }
            try {
              const parsed = JSON.parse(dataStr);
              const delta = parsed.choices?.[0]?.delta?.content || parsed.choices?.[0]?.text || '';
              if (delta) onChunk(delta);
            } catch (err) {
              // Ignore non-json chunks
            }
          }
        }
      }
      onDone();
      return true;
    } catch (err) {
      onError(err);
      return false;
    }
  }

  async function handleSubmit(prompt) {
    appendMessage('user', prompt);
    tokenStreamBox.innerHTML = '<span style="color: var(--primary);">Connecting to inference engine...</span>';

    const startTime = performance.now();
    const asstMsg = appendMessage('assistant', '');
    tokenStreamBox.innerHTML = '';

    let ttftRecorded = false;

    function handleChunk(tokenText) {
      if (!ttftRecorded) {
        const ttft = (performance.now() - startTime).toFixed(1);
        ttftDisplay.textContent = `${ttft} ms`;
        ttftRecorded = true;
      }
      asstMsg.textContent += tokenText;
      chatMessages.scrollTop = chatMessages.scrollHeight;

      const chunk = document.createElement('div');
      chunk.textContent = `data: {"delta": "${tokenText.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"}`;
      tokenStreamBox.appendChild(chunk);
      tokenStreamBox.scrollTop = tokenStreamBox.scrollHeight;
    }

    function handleDone() {
      const doneChunk = document.createElement('div');
      doneChunk.style.color = 'var(--success)';
      doneChunk.textContent = 'data: [DONE]';
      tokenStreamBox.appendChild(doneChunk);
      tokenStreamBox.scrollTop = tokenStreamBox.scrollHeight;
    }

    // Try live API first (if user runs `python scripts/serve.py`)
    const success = await queryLiveApi(
      prompt,
      handleChunk,
      handleDone,
      async (err) => {
        // Fallback to local simulator so user always gets an immediate answer
        const fallbackText = determineResponse(prompt);
        const words = fallbackText.split(' ');
        for (let i = 0; i < words.length; i++) {
          await new Promise(r => setTimeout(r, 30));
          handleChunk(words[i] + ' ');
        }
        handleDone();
      }
    );
  }
});
