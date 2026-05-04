from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from services.llm import embedding_model
import pickle

# Load vectorstore
vectorstore = Chroma(
    persist_directory="data/chroma_db",
    embedding_function=embedding_model
)

# MMR retriever
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5}
)

# Load all chunks
with open("data/all_chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)


def get_doc_topic(doc):
    meta = doc.metadata
    return (
        meta.get("topic", "") or
        meta.get("metadata_topic", "")
    ).strip().lower()


def get_hybrid_retriever(topic=None, k=5):
    bm25_docs = all_chunks

    if topic:
        topic_lower = topic.strip().lower()
        bm25_docs = [
            doc for doc in all_chunks
            if get_doc_topic(doc) == topic_lower
        ]

    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = k

    semantic_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k}
    )

    return EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.2, 0.8]
    )
