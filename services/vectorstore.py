from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
import pickle

from functools import lru_cache

_topic_list = None

def get_all_chunks():
    global _all_chunks
    if _all_chunks is None:
        with open("data/all_chunks.pkl", "rb") as f:
            _all_chunks = pickle.load(f)
    return _all_chunks

def get_topic_list():
    global _topic_list
    if _topic_list is None:
        all_chunks = get_all_chunks()
        seen = set()
        _topic_list = []
        for chunk in all_chunks:
            topic = chunk.metadata.get("topic") or chunk.metadata.get("metadata_topic")
            if topic and topic not in seen:
                seen.add(topic)
                _topic_list.append(topic)
    return _topic_list

# Cache BM25 index per topic — built once, reused forever
_bm25_cache: dict = {}


mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5}
)

def get_hybrid_retriever(topic=None, k=5):
    cache_key = topic.strip().lower() if topic else "__all__"
    
    if cache_key not in _bm25_cache:
        all_docs = get_all_chunks()
        if topic:
            topic_lower = topic.strip().lower()
            filtered = [d for d in all_docs if get_doc_topic(d) == topic_lower]
        else:
            filtered = all_docs
        
        # Build BM25 index once and cache it
        bm25_retriever = BM25Retriever.from_documents(filtered)
        bm25_retriever.k = k
        _bm25_cache[cache_key] = bm25_retriever
    
    bm25_retriever = _bm25_cache[cache_key]

    semantic_retriever = get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            # Add topic filter to semantic search too
            **({"filter": {"topic": topic}} if topic else {})
        }
    )

    return EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.2, 0.8]
    )
