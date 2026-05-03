import json
from openai import OpenAI
from keys import FLUIQ_API_KEY
from fluiq import instrument, trace

instrument(api_key=FLUIQ_API_KEY)

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    }
]


def fake_weather(city: str) -> str:
    return json.dumps({"city": city, "temp_c": 21, "condition": "sunny"})


@trace
def run():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Paris?"},
    ]

    first = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )

    choice = first.choices[0]
    tool_calls = choice.message.tool_calls or []

    messages.append({
        "role": "assistant",
        "content": choice.message.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ],
    })

    for tc in tool_calls:
        args = json.loads(tc.function.arguments or "{}")
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": fake_weather(args.get("city", "")),
        })

    second = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )

    return second.choices[0].message.content


print(run())
