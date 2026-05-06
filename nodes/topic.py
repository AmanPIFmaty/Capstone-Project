import langgraph
from services.state import ChatState
from langfuse import observe
from services.llm import llm
from services.topic_detector import detect_topic_from_query, topic_list


@observe()
def topic_node(state: ChatState):
    query = state["query"]

    topics = detect_topic_from_query(query, topic_list, llm)

    return {
        "topics": topics
    }

