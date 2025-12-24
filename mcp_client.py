"""MCP Client for connecting to the customer support MCP server."""
import httpx
from typing import Any


class MCPClient:
    """Client for interacting with the MCP server via JSON-RPC over HTTP."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.tools: list[dict] = []
    
    async def initialize(self) -> None:
        """Fetch available tools from the MCP server."""
        self.tools = await self.list_tools()
    
    async def list_tools(self) -> list[dict]:
        """Get list of available tools from MCP server."""
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.post(
                self.server_url,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {}
                },
                timeout=30.0
            )
            result = response.json()
            if "result" in result and "tools" in result["result"]:
                return result["result"]["tools"]
            return []
    
    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server."""
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.post(
                self.server_url,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    }
                },
                timeout=30.0
            )
            result = response.json()
            if "result" in result:
                content = result["result"].get("content", [])
                if content and len(content) > 0:
                    return content[0].get("text", str(result["result"]))
            if "error" in result:
                return f"Error: {result['error'].get('message', 'Unknown error')}"
            return str(result)
    
    def get_tools_for_llm(self) -> list[dict]:
        """Convert MCP tools to a format suitable for LLM function calling."""
        llm_tools = []
        for tool in self.tools:
            llm_tools.append({
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool.get("inputSchema", {"type": "object", "properties": {}})
            })
        return llm_tools

