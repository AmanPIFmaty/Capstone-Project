from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embedding_model

qwenn_llm = ChatGroq(
    model="qwen-qwq-32b",
    api_key=os.getenv("GROQ_API_KEY")
)
