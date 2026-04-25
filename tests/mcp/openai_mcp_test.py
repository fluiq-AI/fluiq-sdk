from openai import OpenAI
from dotenv import load_dotenv
from fluiq import instrument, trace

instrument(api_key="YOUR_KEY")
load_dotenv()

client = OpenAI()


@trace
def run():
    resp = client.responses.create(
        model="gpt-4.1",
        tools=[
            {
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "require_approval": "never",
            }
        ],
        input="What transport protocols does the 2025-03-26 MCP spec support?",
    )
    return resp


print(run())
