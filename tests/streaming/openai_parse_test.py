from openai import OpenAI
from pydantic import BaseModel
from ..keys import FLUIQ_API_KEY
from fluiq import instrument

instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")


class Capital(BaseModel):
    country: str
    capital: str


client = OpenAI()

response = client.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Extract the country and its capital."},
        {"role": "user", "content": "France's capital is Paris."},
    ],
    response_format=Capital,
)

print(response.choices[0].message.parsed)
