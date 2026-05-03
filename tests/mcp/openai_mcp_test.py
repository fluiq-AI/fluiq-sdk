from openai import OpenAI
from keys import FLUIQ_API_KEY
from fluiq import instrument, trace

instrument(api_key=FLUIQ_API_KEY)

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
