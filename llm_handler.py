"""LLM Handler using OpenAI or OpenRouter for customer support chatbot."""
import os
import json
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv
from mcp_client import MCPClient

# Load environment variables from project root .env file
ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(ENV_FILE, override=True)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
USE_OPEN_ROUTER = os.getenv("USE_OPEN_ROUTER", "false").lower() == "true"

# Initialize async client
if USE_OPEN_ROUTER:
    client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    # OpenRouter requires provider prefix for OpenAI models
    if not AI_MODEL.startswith("openai/"):
        AI_MODEL = f"openai/{AI_MODEL}"
else:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)


SYSTEM_PROMPT = """You are a helpful customer support assistant for TechStore, a company that sells computer products including:
- Computers (desktops, laptops, workstations, gaming PCs)
- Monitors (various sizes, 4K, ultrawide, curved)
- Printers (laser, inkjet, 3D printers, label printers)
- Accessories (keyboards, mice, webcams, headsets)
- Networking equipment (routers, switches, modems)

You have access to tools to help customers:
- Search and browse products
- Get product details and pricing
- Look up customer information
- View and manage orders

Be friendly, helpful, and concise. When customers ask about products, use the available tools to provide accurate information.
If a customer wants to place an order, you'll need their customer ID and verification (email + PIN).

Always be professional and aim to resolve customer inquiries efficiently."""


class LLMHandler:
    """Handles LLM interactions with OpenAI or OpenRouter."""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.conversation_history: list[dict] = []
    
    def _build_tools_description(self) -> str:
        """Build a description of available tools for the LLM."""
        tools = self.mcp_client.get_tools_for_llm()
        tools_text = "\n\nAvailable tools:\n"
        for tool in tools:
            params = tool.get("parameters", {}).get("properties", {})
            required = tool.get("parameters", {}).get("required", [])
            param_str = ", ".join([
                f"{k}: {v.get('type', 'any')}" + (" (required)" if k in required else " (optional)")
                for k, v in params.items()
            ])
            tools_text += f"\n- {tool['name']}({param_str})\n  {tool['description'][:200]}...\n"
        
        tools_text += """
To use a tool, respond with a JSON object in this exact format:
{"tool": "tool_name", "arguments": {"param1": "value1"}}

Only respond with the JSON when you need to use a tool. Otherwise, respond normally to the user."""
        return tools_text
    
    async def process_message(self, user_message: str) -> str:
        """Process a user message and return the response."""
        # Build the full prompt
        tools_desc = self._build_tools_description()
        full_system = SYSTEM_PROMPT + tools_desc
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Build messages for OpenAI
        messages = [{"role": "system", "content": full_system}]
        # Keep last 10 messages for context
        messages.extend(self.conversation_history[-10:])
        
        # Get response from LLM
        completion = await client.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
            temperature=0.7
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        # Check if the response is a tool call
        tool_result = await self._try_parse_tool_call(response_text)
        if tool_result:
            # Execute the tool and get the result
            tool_name, arguments, result = tool_result
            
            # Ask LLM to formulate a response with the tool result
            tool_response_messages = messages + [
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": f"""The tool '{tool_name}' was called with arguments {json.dumps(arguments)} and returned:

{result}

Please provide a helpful response to the user based on this information. Be concise and friendly. Do not include any JSON in your response."""}
            ]
            
            followup = await client.chat.completions.create(
                model=AI_MODEL,
                messages=tool_response_messages,
                temperature=0.7
            )
            final_response = followup.choices[0].message.content.strip()
            
            # Clean up any tool call JSON from the response
            final_response = self._clean_response(final_response)
            
            self.conversation_history.append({"role": "assistant", "content": final_response})
            return final_response
        
        # Clean up response
        response_text = self._clean_response(response_text)
        self.conversation_history.append({"role": "assistant", "content": response_text})
        return response_text
    
    async def _try_parse_tool_call(self, text: str) -> tuple[str, dict, str] | None:
        """Try to parse a tool call from the response text."""
        # Look for JSON in the response
        try:
            # Try to find JSON object in the text
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                # Fix Python-style booleans/None to JSON-style
                json_str = json_str.replace(': True', ': true')
                json_str = json_str.replace(':True', ':true')
                json_str = json_str.replace(': False', ': false')
                json_str = json_str.replace(':False', ':false')
                json_str = json_str.replace(': None', ': null')
                json_str = json_str.replace(':None', ':null')
                data = json.loads(json_str)
                if "tool" in data and "arguments" in data:
                    tool_name = data["tool"]
                    arguments = data["arguments"]
                    # Execute the tool
                    result = await self.mcp_client.call_tool(tool_name, arguments)
                    return (tool_name, arguments, result)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Tool call parse error: {e}")
            pass
        return None
    
    def _clean_response(self, text: str) -> str:
        """Remove any tool call JSON from the response."""
        # If the response is just a JSON tool call, return empty
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                data = json.loads(json_str)
                if "tool" in data and "arguments" in data:
                    # Remove the JSON part
                    before = text[:start].strip()
                    after = text[end:].strip()
                    return (before + " " + after).strip() or "Let me look that up for you..."
        except (json.JSONDecodeError, KeyError):
            pass
        return text
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
