from sentence_transformers import CrossEncoder
from langfuse import observe

reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_documents(query, docs, top_k=3):
    pairs = [(query, doc.page_content) for doc in docs]

    scores = reranker_model.predict(pairs)

    scored_docs = list(zip(docs, scores))

    scored_docs = sorted(
        scored_docs,
        key=lambda x: x[1],
        reverse=True
    )
    final_docs = [doc for doc, _ in scored_docs[:top_k]]

    return final_docs

@observe()
def reranker_node(state):
    query = state["query"]
    docs = state["docs"]

    reranked_docs = rerank_documents(query, docs, top_k=3)
    context = " ".join([doc.page_content for doc in reranked_docs])

    return {
        "docs": reranked_docs,
        "context": context
    }
