"""FastAPI Customer Support Chatbot with MCP integration."""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from mcp_client import MCPClient
from llm_handler import LLMHandler

# Load environment variables from project root .env file
ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(ENV_FILE, override=True)

# Configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://vipfapwm3x.us-east-1.awsapprunner.com/mcp")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Global instances
mcp_client: MCPClient | None = None
llm_handler: LLMHandler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize MCP client and LLM handler on startup."""
    global mcp_client, llm_handler
    
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set. Please set it in .env file.")
    
    # Initialize MCP client
    mcp_client = MCPClient(MCP_SERVER_URL)
    await mcp_client.initialize()
    print(f"MCP Client initialized with {len(mcp_client.tools)} tools")
    
    # Initialize LLM handler
    llm_handler = LLMHandler(mcp_client)
    print("LLM Handler initialized with OpenAI")
    
    yield
    
    # Cleanup
    print("Shutting down...")


app = FastAPI(
    title="TechStore Customer Support",
    description="AI-powered customer support chatbot for computer products",
    lifespan=lifespan
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat UI."""
    return HTML_TEMPLATE


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return the response."""
    if not llm_handler:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        response = await llm_handler.process_message(request.message)
        return ChatResponse(response=response)
    except Exception as e:
        print(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear_history():
    """Clear the conversation history."""
    if llm_handler:
        llm_handler.clear_history()
    return {"status": "ok"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mcp_connected": mcp_client is not None and len(mcp_client.tools) > 0,
        "tools_count": len(mcp_client.tools) if mcp_client else 0
    }


# Simple HTML template for the chat UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TechStore Support</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        header {
            background: #16213e;
            padding: 1rem 2rem;
            border-bottom: 2px solid #0f3460;
        }
        
        h1 {
            font-size: 1.4rem;
            color: #e94560;
        }
        
        h1 span {
            color: #888;
            font-weight: normal;
            font-size: 0.9rem;
        }
        
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .message {
            max-width: 80%;
            padding: 0.8rem 1rem;
            border-radius: 12px;
            line-height: 1.6;
        }
        
        .message.user {
            background: #0f3460;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }
        
        .message.assistant {
            background: #16213e;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
            border: 1px solid #0f3460;
        }
        
        .message.system {
            background: #2a2a4a;
            align-self: center;
            font-size: 0.9rem;
            color: #888;
        }
        
        /* Formatted content styles */
        .message.assistant h1, .message.assistant h2, .message.assistant h3,
        .message.assistant h4, .message.assistant h5, .message.assistant h6 {
            color: #e94560;
            margin: 0.8rem 0 0.4rem 0;
            font-weight: 600;
        }
        
        .message.assistant h1:first-child, .message.assistant h2:first-child,
        .message.assistant h3:first-child {
            margin-top: 0;
        }
        
        .message.assistant h1 { font-size: 1.4rem; }
        .message.assistant h2 { font-size: 1.25rem; }
        .message.assistant h3 { font-size: 1.1rem; }
        .message.assistant h4 { font-size: 1rem; }
        .message.assistant h5 { font-size: 0.95rem; }
        .message.assistant h6 { font-size: 0.9rem; color: #c73e54; }
        
        .message.assistant strong, .message.assistant b {
            color: #e94560;
            font-weight: 600;
        }
        
        .message.assistant ul, .message.assistant ol {
            margin: 0.5rem 0 0.5rem 1.5rem;
        }
        
        .message.assistant ul.numbered-list {
            list-style: none;
            margin-left: 0;
            padding-left: 0;
        }
        
        .message.assistant li {
            margin: 0.4rem 0;
        }
        
        .message.assistant p {
            margin: 0.5rem 0;
        }
        
        .message.assistant p:first-child {
            margin-top: 0;
        }
        
        .message.assistant p:last-child {
            margin-bottom: 0;
        }
        
        .message.assistant code {
            background: #0a0a1a;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.9em;
            color: #4fc3f7;
        }
        
        .message.assistant .order-item {
            background: #0f3460;
            border-radius: 8px;
            padding: 0.6rem 0.8rem;
            margin: 0.5rem 0;
            border-left: 3px solid #e94560;
        }
        
        .message.assistant .order-id {
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.85em;
            color: #4fc3f7;
        }
        
        .message.assistant .status {
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 600;
        }
        
        .message.assistant .status.fulfilled { background: #1b5e20; color: #a5d6a7; }
        .message.assistant .status.submitted { background: #0d47a1; color: #90caf9; }
        .message.assistant .status.pending { background: #e65100; color: #ffcc80; }
        .message.assistant .status.cancelled { background: #b71c1c; color: #ef9a9a; }
        
        .message.assistant .price {
            color: #81c784;
            font-weight: 600;
        }
        
        .input-area {
            background: #16213e;
            padding: 1rem;
            border-top: 2px solid #0f3460;
            display: flex;
            gap: 0.5rem;
        }
        
        #message-input {
            flex: 1;
            padding: 0.8rem 1rem;
            border: 1px solid #0f3460;
            border-radius: 8px;
            background: #1a1a2e;
            color: #eee;
            font-size: 1rem;
        }
        
        #message-input:focus {
            outline: none;
            border-color: #e94560;
        }
        
        button {
            padding: 0.8rem 1.5rem;
            background: #e94560;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            transition: background 0.2s;
        }
        
        button:hover {
            background: #c73e54;
        }
        
        button:disabled {
            background: #555;
            cursor: not-allowed;
        }
        
        button.secondary {
            background: #0f3460;
        }
        
        button.secondary:hover {
            background: #1a4a7a;
        }
        
        .typing {
            display: none;
            padding: 0.5rem 1rem;
            color: #888;
            font-style: italic;
        }
        
        .typing.active {
            display: block;
        }
        
        pre {
            background: #0a0a1a;
            padding: 0.5rem;
            border-radius: 4px;
            overflow-x: auto;
            margin: 0.5rem 0;
        }
    </style>
</head>
<body>
    <header>
        <h1>TechStore Support <span>| AI Customer Assistant</span></h1>
    </header>
    
    <div class="chat-container" id="chat-container">
        <div class="message system">
            Welcome! I'm your TechStore support assistant. I can help you with products, orders, and account questions.
        </div>
    </div>
    
    <div class="typing" id="typing-indicator">Assistant is typing...</div>
    
    <div class="input-area">
        <input type="text" id="message-input" placeholder="Type your message..." autocomplete="off">
        <button id="send-btn" onclick="sendMessage()">Send</button>
        <button class="secondary" onclick="clearChat()">Clear</button>
    </div>
    
    <script>
        const chatContainer = document.getElementById('chat-container');
        const messageInput = document.getElementById('message-input');
        const sendBtn = document.getElementById('send-btn');
        const typingIndicator = document.getElementById('typing-indicator');
        
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Convert markdown-style text to HTML
        function formatMessage(text) {
            // Escape HTML first to prevent XSS
            let html = text
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
            
            // Convert markdown to HTML
            // Headers: # H1, ## H2, ### H3, etc.
            html = html.replace(/^######\\s+(.+)$/gm, '<h6>$1</h6>');
            html = html.replace(/^#####\\s+(.+)$/gm, '<h5>$1</h5>');
            html = html.replace(/^####\\s+(.+)$/gm, '<h4>$1</h4>');
            html = html.replace(/^###\\s+(.+)$/gm, '<h3>$1</h3>');
            html = html.replace(/^##\\s+(.+)$/gm, '<h2>$1</h2>');
            html = html.replace(/^#\\s+(.+)$/gm, '<h1>$1</h1>');
            
            // Bold: **text** or __text__
            html = html.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
            html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');
            
            // Inline code: `code`
            html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
            
            // Format order IDs (UUID pattern)
            html = html.replace(/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/gi, 
                '<span class="order-id">$1</span>');
            
            // Format prices ($X,XXX.XX)
            html = html.replace(/(\\$[\\d,]+\\.\\d{2})/g, '<span class="price">$1</span>');
            
            // Format status words
            html = html.replace(/\\b(Fulfilled)\\b/gi, '<span class="status fulfilled">$1</span>');
            html = html.replace(/\\b(Submitted)\\b/gi, '<span class="status submitted">$1</span>');
            html = html.replace(/\\b(Pending)\\b/gi, '<span class="status pending">$1</span>');
            html = html.replace(/\\b(Cancelled|Canceled)\\b/gi, '<span class="status cancelled">$1</span>');
            
            // Numbered lists: 1. item (use <ul> to avoid double numbering)
            html = html.replace(/^(\\d+)\\.\\s+(.+)$/gm, '<li><strong>$1.</strong> $2</li>');
            
            // Wrap consecutive <li> in <ul> (not <ol> to avoid auto-numbering)
            html = html.replace(/(<li>.*<\\/li>\\n?)+/g, '<ul class="numbered-list">$&</ul>');
            
            // Convert line breaks to paragraphs (but not inside lists or headers)
            const parts = html.split(/(<ul.*?<\\/ul>|<h[1-6]>.*?<\\/h[1-6]>)/gs);
            html = parts.map(part => {
                if (part.startsWith('<ul') || part.startsWith('<h')) return part;
                // Convert double newlines to paragraph breaks
                return part.split(/\\n\\n+/).map(p => p.trim()).filter(p => p).map(p => `<p>${p}</p>`).join('');
            }).join('');
            
            // Convert remaining single newlines to <br> within paragraphs
            html = html.replace(/([^>])\\n([^<])/g, '$1<br>$2');
            
            return html;
        }
        
        function addMessage(content, role) {
            const div = document.createElement('div');
            div.className = 'message ' + role;
            
            if (role === 'assistant') {
                // Format assistant messages with HTML
                div.innerHTML = formatMessage(content);
            } else {
                // Keep user messages as plain text
                div.textContent = content;
            }
            
            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            addMessage(message, 'user');
            messageInput.value = '';
            sendBtn.disabled = true;
            typingIndicator.classList.add('active');
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    addMessage(data.response, 'assistant');
                } else {
                    addMessage('Error: ' + (data.detail || 'Something went wrong'), 'system');
                }
            } catch (err) {
                addMessage('Error: Could not connect to server', 'system');
            } finally {
                sendBtn.disabled = false;
                typingIndicator.classList.remove('active');
                messageInput.focus();
            }
        }
        
        async function clearChat() {
            await fetch('/clear', { method: 'POST' });
            chatContainer.innerHTML = '<div class="message system">Chat cleared. How can I help you?</div>';
        }
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
