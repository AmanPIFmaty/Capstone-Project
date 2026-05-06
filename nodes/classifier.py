import langgraph
from services.state import ChatState
from services.llm import llm
from services.classifier import classify_query
from langfuse import observe


@observe()
def classify_node(state: ChatState):
    query = state.get("final_query")
    query_type = classify_query(query, llm)

    return {
        "final_query": query,
        "query_type": query_type
    }


