import anthropic
import fluiq
from dotenv import load_dotenv

fluiq.instrument(api_key="YOUR_KEY")
load_dotenv()

client = anthropic.Anthropic()

msg = client.beta.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    mcp_servers=[
        {
            "type": "url",
            "url": "https://mcp.deepwiki.com/mcp",
            "name": "deepwiki",
        }
    ],
    messages=[
        {
            "role": "user",
            "content": "What transport protocols does the MCP spec support?",
        }
    ],
    betas=["mcp-client-2025-04-04"],
)

print(msg)
