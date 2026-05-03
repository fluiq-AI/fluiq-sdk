import json
import anthropic
import fluiq
from keys import FLUIQ_API_KEY

fluiq.instrument(api_key=FLUIQ_API_KEY)

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
        },
    }
]


def fake_weather(city: str) -> str:
    return json.dumps({"city": city, "temp_c": 21, "condition": "sunny"})


@fluiq.trace
def run():
    messages = [
        {"role": "user", "content": "What's the weather in Paris?"},
    ]

    first = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        tools=tools,
        messages=messages,
    )

    messages.append({"role": "assistant", "content": first.content})

    tool_results = []
    for block in first.content or []:
        if getattr(block, "type", None) == "tool_use":
            args = block.input or {}
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": fake_weather(args.get("city", "")),
            })

    if tool_results:
        messages.append({"role": "user", "content": tool_results})

    second = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        tools=tools,
        messages=messages,
    )

    return second


print(run())
