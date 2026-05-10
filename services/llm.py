from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1" 
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



qwenn_llm = ChatOllama(
    model="qwen2.5:3b"     
)
