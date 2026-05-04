import langgraph
from services.state import ChatState
from services.llm import llm
from services.classifier import classify_query


def classify_node(state: ChatState):
    query = state["messages"][-1].content

    query_type = classify_query(query, llm)

    return {
        "query": query,
        "query_type": query_type
    }
    
