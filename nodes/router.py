import langgraph
from services.state import ChatState


def route_query(state: ChatState):
    return state["query_type"]

def route_input(state):
    if state.get("query_type") == "block":
        return "block"                      

    if state.get("image_bytes") and not state.get("query"):
        return "image"

    return "text"
