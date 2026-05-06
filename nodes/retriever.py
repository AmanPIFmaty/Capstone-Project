import langgraph
from langfuse import observe
from services.state import ChatState
from services.vectorstore import (
    mmr_retriever,
    get_hybrid_retriever
)

@observe()
def retriever_node(state: ChatState):
    query = state.get("final_query")
    topics = state.get("topics", [])
    if topics:
        docs = get_hybrid_retriever(topic=topics[0], k=5).invoke(query)
    else:
        docs = mmr_retriever.invoke(query)
        

    return {
        "docs": docs
    }





