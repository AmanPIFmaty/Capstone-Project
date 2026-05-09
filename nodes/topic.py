from services.topic_detector import detect_topic_from_query, get_topic_list
import langgraph
from services.state import ChatState
from langfuse import observe

@observe()
def topic_node(state: ChatState):
    query = state.get("final_query")
    topics = detect_topic_from_query(query, get_topic_list())
    return {"topics": topics}
