import os
from ..keys import FLUIQ_API_KEY
import fluiq

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")

from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

get_weather = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Get the current weather for a city.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "city": types.Schema(type="STRING"),
                },
                required=["city"],
            ),
        )
    ]
)


def fake_weather(city: str) -> dict:
    return {"city": city, "temp_c": 21, "condition": "sunny"}


@fluiq.trace
def run():
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="What's the weather in Paris?")],
        )
    ]

    first = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=[get_weather]),
    )

    candidate = (first.candidates or [None])[0]
    parts = getattr(getattr(candidate, "content", None), "parts", None) or []

    contents.append(types.Content(role="model", parts=parts))

    response_parts = []
    for part in parts:
        fc = getattr(part, "function_call", None)
        if fc and getattr(fc, "name", None):
            args = dict(fc.args or {})
            response_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response=fake_weather(args.get("city", "")),
                )
            )

    if response_parts:
        contents.append(types.Content(role="user", parts=response_parts))

    second = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=[get_weather]),
    )

    if second.text:
        return second.text

    second_parts = getattr(getattr((second.candidates or [None])[0], "content", None), "parts", None) or []
    return [p.model_dump(mode="json", exclude_none=True) for p in second_parts]


print(run())
