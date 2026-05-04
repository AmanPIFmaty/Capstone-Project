from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings

llm = Ollama(model="llama3.1")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)