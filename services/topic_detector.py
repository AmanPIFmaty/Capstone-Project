import numpy as np
import pickle
from services.llm import get_embedding_model

# ✅ declare all module-level state at the top
_all_chunks = None
_topic_list = None
_topic_embeddings = None
_topic_list_cache = None


def get_all_chunks():
    global _all_chunks
    if _all_chunks is None:
        with open("data/all_chunks.pkl", "rb") as f:
            _all_chunks = pickle.load(f)
    return _all_chunks


def get_topic_list():
    global _topic_list
    if _topic_list is None:
        seen = set()
        _topic_list = []
        for chunk in get_all_chunks():
            topic = chunk.metadata.get("topic") or chunk.metadata.get("metadata_topic")
            if topic and topic not in seen:
                seen.add(topic)
                _topic_list.append(topic)
    return _topic_list


def _get_topic_embeddings(topic_list):
    global _topic_embeddings, _topic_list_cache
    if _topic_embeddings is None or _topic_list_cache != topic_list:
        embedder = get_embedding_model()
        _topic_embeddings = embedder.embed_documents(topic_list)
        _topic_list_cache = topic_list
    return _topic_embeddings


def get_doc_topic(doc):
    meta = doc.metadata
    return (
        meta.get("topic", "") or
        meta.get("metadata_topic", "")
    ).strip().lower()


def detect_topic_query(query: str, topic_list: list, llm=None) -> list:
    if not topic_list:
        return []

    embedder = get_embedding_model()
    query_emb = np.array(embedder.embed_query(query))
    topic_embs = np.array(_get_topic_embeddings(topic_list))

    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    topic_norms = topic_embs / (np.linalg.norm(topic_embs, axis=1, keepdims=True) + 1e-9)
    scores = topic_norms @ query_norm

    top_indices = np.argsort(scores)[::-1][:3]
    return [topic_list[i] for i in top_indices if scores[i] > 0.3]


def detect_topic_from_query(query: str, topic_list: list, llm=None) -> list:
    if not topic_list:
        return []

    embedder = get_embedding_model()
    query_emb = np.array(embedder.embed_query(query))
    topic_embs = np.array(_get_topic_embeddings(topic_list))

    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    topic_norms = topic_embs / (np.linalg.norm(topic_embs, axis=1, keepdims=True) + 1e-9)
    scores = topic_norms @ query_norm

    top_indices = np.argsort(scores)[::-1][:3]
    
    # Lower threshold to 0.2, but always return at least the top 1
    results = [topic_list[i] for i in top_indices if scores[i] > 0.2]
    if not results:
        results = [topic_list[top_indices[0]]]  # fallback: best match regardless of score
    
    return results
