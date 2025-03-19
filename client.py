import asyncio
import sys
from typing import Optional
from ollama import Client
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClient:
  def __init__(self):
    """Initialize the MCP client."""
    self.session: Optional[ClientSession] = None
    self.exit_stack = AsyncExitStack()
    self.ollama = Client()

  async def connect_to_server(self, server_script_path: str):
    """Connect to an MCP server.
    
    Args:
      server_script_path (str): Path to the server script (.py or .js).
    """
    is_python = server_script_path.endswith('.py')
    is_js = server_script_path.endswith('.js')
    if not (is_python or is_js):
      raise ValueError("Server script must be a .py or .js file")
        
    command = "python" if is_python else "node"
    server_params = StdioServerParameters(
      command=command,
      args=[server_script_path],
      env=None
    )
    
    stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
    self.stdio, self.write = stdio_transport
    self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
    
    await self.session.initialize()

    # List available tools
    response = await self.session.list_tools()
    tools = response.tools
    print("\nConnected to server with tools:", [tool.name for tool in tools])

  async def process_query(self, query: str) -> str:
    """Process a user query using Llama and available tools."""
    messages = [
      {
        "role": "system",
        "content": (
          "Answer the user's question concisely, and only use tools if the user asks something that the tools you have would be helpful."
        )
      },
      {"role": "user", "content": query}
    ]

    response = await self.session.list_tools()

    available_tools = [
      {
        "type": "function",
        "function": {
          "name": tool.name,
          "description": tool.description,
          "parameters": tool.inputSchema
        },
      } for tool in response.tools
    ]

    # Initial Llama API call
    response = self.ollama.chat(
      model="llama3.2:1b",
      messages=messages,
      tools=available_tools,
    )
    if not response or not response.message:
      return "I wasn't able to process your query, please try again!"
    
    if response.message.content:
      return response.message.content
    
    # Process response and handle tool calls
    final_text = []

    use_of_tools = []
    if response.message.tool_calls:
      for tool in response.message.tool_calls:
        tool_name = tool.function.name
        tool_args = tool.function.arguments

        if tool_name not in [t["function"]["name"] for t in available_tools]:
          final_text.append(f"[Error: Tool '{tool_name}' does not exist!]")
          # we still continue because AI can give some response even without this tool
          continue

        # Execute tool call
        result = await self.session.call_tool(tool_name, tool_args)
        use_of_tools.append({"call": tool_name, "result": result})
        final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

        messages.append({"role": "user", "content": result.content[0].text})

      # Get next response from Llama
      if use_of_tools:
        response = self.ollama.chat(
          model="llama3.2:1b",
          messages=messages,
        )
        
        final_text.append(response.message.content)

    return "\n".join(final_text)


  async def chat_loop(self):
    """Run an interactive chat loop."""
    print("\nMCP Client Started!")
    print("Type your queries or 'quit' to exit.")
    
    while True:
      try:
        query = input("\nQuery: ").strip()
        
        if query.lower() == 'quit':
          break
              
        response = await self.process_query(query)
        print(response)
              
      except Exception as e:
        print(f"\nError: {str(e)}")
  
  async def cleanup(self):
    """Clean up resources."""
    await self.exit_stack.aclose()


async def main():
  """Main function to start the client."""
  if len(sys.argv) < 2:
    print("Usage: python client.py <path_to_server_script>")
    sys.exit(1)
      
  client = MCPClient()
  try:
    await client.connect_to_server(sys.argv[1])
    await client.chat_loop()
  finally:
    await client.cleanup()


if __name__ == "__main__":
  asyncio.run(main())