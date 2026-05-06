from langchain_core.messages import AIMessage, HumanMessage
from langfuse import observe

ABUSIVE_WORDS = [
    "fuck", "shit", "bitch", "idiot", "stupid", "asshole",
    "bastard", "moron", "dumb", "loser","porn","nudes"
]

def is_abusive(text: str) -> bool:
    return any(word in text.lower() for word in ABUSIVE_WORDS)

@observe()
def input_guard_node(state):
    human_messages = [
        m for m in state["messages"]
        if isinstance(m, HumanMessage) or m.type == "human"
    ]
    #query = human_messages[-1].content if human_messages else ""
    query = state.get("query", "")
    if is_abusive(query):
        return {
            "messages": [AIMessage(content="⚠️ Please use respectful language. I'm here to help.")],
            "query_type": "block",
            "query": "",
            "image_bytes": None
        }

    return {
        "query": query,
        "query_type": None,
        "image_bytes": state.get("image_bytes")
    }
