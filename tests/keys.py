import os
from dotenv import load_dotenv

load_dotenv()

FLUIQ_API_KEY = os.getenv("FLUIQ_API_KEY")
GEMINI_KEY=os.getenv("GEMINI_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY")
PERPLEXITY_API_KEY=os.getenv("PERPLEXITY_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")   # e.g. https://my-index-xyz.svc.pinecone.io
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")             # None for local
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")         # None for local