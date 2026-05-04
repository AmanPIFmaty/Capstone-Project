# %% [markdown]
# ### NODE BREAKDOWN

# %%
from typing import TypedDict, Annotated
from langchain_core.messages import AIMessage,HumanMessage,BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langchain_core.documents import Document

# %%
class ChatState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    query_type: str
    topics: list[str]
    docs: list[Document]
    context: str
    query: str

# %%
from nodes.classifier import classify_node
from nodes.topic import topic_node
from nodes.retriever import retriever_node
from nodes.generator import generator_node
from nodes.chat import chat_node
from nodes.router import route_query


def build_graph():
    workflow = StateGraph(ChatState)

    workflow.add_node("classifier", classify_node)
    workflow.add_node("topic", topic_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("chat", chat_node)

    workflow.add_edge(START, "classifier")

    workflow.add_conditional_edges(
        "classifier",
        route_query,
        {
            "rag": "topic",
            "chat": "chat"
        }
    )

    workflow.add_edge("topic", "retriever")
    workflow.add_edge("retriever", "generator")

    workflow.add_edge("generator", END)
    workflow.add_edge("chat", END)
    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


# %%
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

chatbot = build_graph()

config = {"configurable": {"thread_id": "th-1"}}

while True:
    query = input("User: ")

    if query.lower() == "quit":
        break

    response = chatbot.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config
    )

    print("AI:", response["messages"][-1].content)

# %%
