import logging
logging.basicConfig(level=logging.WARNING)

from openai import OpenAI
from ..keys import FLUIQ_API_KEY
import fluiq

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.secure()

client = OpenAI()

def res():

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Book tickets to Paris. My name is Bo Lan and my phone number is 8565854215."}
        ]
    )

    return response.choices[0].message.content

print(res())