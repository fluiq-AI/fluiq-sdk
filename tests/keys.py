import os
from dotenv import load_dotenv

load_dotenv()

FLUIQ_API_KEY = os.getenv("FLUIQ_API_KEY") 
GEMINI_KEY=os.getenv("GEMINI_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY")
PERPLEXITY_API_KEY=os.getenv("PERPLEXITY_API_KEY")