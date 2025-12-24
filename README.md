---
title: TechStore Customer Support
emoji: 🛒
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: false
license: mit
---

# MCP-Chatbot

## Example of a TechStore Customer Support Chatbot

A simple AI-powered customer support chatbot prototype using MCP (Model Context Protocol) for a computer products company.

## Features

- Chat interface for customer inquiries
- Integration with MCP server for product/order management
- Uses OpenAI (or OpenRouter) for LLM responses
- Real-time product search and information
- Order lookup and management capabilities

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   Copy `.env.example` to `.env` and fill in your API key:
   ```bash
   cp .env.example .env
   ```

3. **Run the server:**
   ```bash
   python main.py
   ```
   
   Or with uvicorn:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Open the chatbot:**
   Navigate to http://localhost:8000

## Available MCP Tools

The chatbot can use these tools to help customers:

- `list_products` - Browse products by category
- `get_product` - Get details for a specific product (by SKU)
- `search_products` - Search products by name/description
- `get_customer` - Look up customer information
- `verify_customer_pin` - Verify customer identity
- `list_orders` - View order history
- `get_order` - Get order details
- `create_order` - Place new orders

## Example Queries

- "What monitors do you have?"
- "Show me gaming desktops under $2000"
- "What's the price of product MON-0056?"
- "Search for wireless keyboards"
- "I need a printer for photos"

## Project Structure

```
mcp_chatbot/
├── main.py          # FastAPI app with embedded UI
├── mcp_client.py    # MCP server client
├── llm_handler.py   # OpenAI/OpenRouter LLM integration
├── requirements.txt # Python dependencies
├── Dockerfile       # Docker config for HF Spaces
├── .env.example     # Environment variables template
├── .gitignore       # Git ignore rules
└── README.md        # This file
```

## Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Docker** as the SDK
3. Clone your Space repo and copy these files into it
4. Go to **Settings → Variables and secrets** and add:
   - `OPENAI_API_KEY` (as a secret)
   - `OPENAI_MODEL` = `gpt-4o-mini`
   - `MCP_SERVER_URL` = `https://vipfapwm3x.us-east-1.awsapprunner.com/mcp`
5. Push to the Space repo and it will auto-deploy

## Tech Stack

- **Backend:** FastAPI
- **LLM:** OpenAI GPT-4o-mini (or any OpenAI/OpenRouter model)
- **Protocol:** MCP (Model Context Protocol)
- **Frontend:** Vanilla HTML/CSS/JS (embedded)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI or OpenRouter API key | (required) |
| `OPENAI_MODEL` | Model to use | `gpt-4o-mini` |
| `USE_OPEN_ROUTER` | Use OpenRouter instead of OpenAI | `false` |
| `MCP_SERVER_URL` | MCP server endpoint | `https://vipfapwm3x.us-east-1.awsapprunner.com/mcp` |
