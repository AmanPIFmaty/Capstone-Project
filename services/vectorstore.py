from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
import pickle

_vectorstore = None
_all_chunks = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        from services.lazy_llm import get_embedding_model
        _vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=get_embedding_model()
        )
    return _vectorstore

def get_all_chunks():
    global _all_chunks
    if _all_chunks is None:
        with open("data/all_chunks.pkl", "rb") as f:
            _all_chunks = pickle.load(f)
    return _all_chunks

def get_doc_topic(doc):
    meta = doc.metadata
    return (
        meta.get("topic", "") or
        meta.get("metadata_topic", "")
    ).strip().lower()

def get_hybrid_retriever(topic=None, k=5):
    bm25_docs = get_all_chunks()  # only loads when called

    if topic:
        topic_lower = topic.strip().lower()
        bm25_docs = [
            doc for doc in bm25_docs
            if get_doc_topic(doc) == topic_lower
        ]

    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = k

    semantic_retriever = get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": k}
    )

    return EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.2, 0.8]
    )
