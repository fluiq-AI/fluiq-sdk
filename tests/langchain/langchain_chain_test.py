from keys import FLUIQ_API_KEY
from fluiq import instrument, trace
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

instrument(api_key=FLUIQ_API_KEY)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers in one short sentence."),
    ("user", "{question}"),
])

chain = prompt | ChatOpenAI(model="gpt-4o") | StrOutputParser()


@trace
def run():
    return chain.invoke({"question": "What is the capital of Germany?"})


print(run())
