import langgraph
from services.state import ChatState
from services.vectorstore import (
    mmr_retriever,
    get_hybrid_retriever
)


def retriever_node(state: ChatState):
    query = state["query"]
    topics = state.get("topics", [])

    if topics:
        docs = get_hybrid_retriever(topic=topics[0], k=5).invoke(query)
    else:
        docs = mmr_retriever.invoke(query)

    return {
        "docs": docs
    }

