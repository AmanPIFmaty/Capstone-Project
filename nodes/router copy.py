import langgraph
from services.state import ChatState


def route_query(state: ChatState):
    return state["query_type"]