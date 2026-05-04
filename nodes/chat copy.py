import langgraph
from services.state import ChatState
from services.llm import llm
from langchain_core.messages import AIMessage


def chat_node(state: ChatState):
    query = state["query"]

    response = llm.invoke(query)

    return {
        "messages": [AIMessage(content=response)]
    }

