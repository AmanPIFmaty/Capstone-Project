from typing import TypedDict, Annotated,Optional
from langchain_core.messages import AIMessage,HumanMessage,BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langchain_core.documents import Document
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

class ChatState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    query_type: str
    topics: list[str]
    docs: list[Document]
    context: str
    query: str
    image_bytes: Optional[bytes]
    ocr_text: str
    final_query: str
    response: str

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


from nodes.classifier import classify_node
from nodes.topic import topic_node
from nodes.retriever import retriever_node
from nodes.generator import generator_node
#from nodes.streaming_generator import generator_node
from nodes.chat import chat_node
#from nodes.streaming_chat import chat_node
from nodes.router import route_query
from langfuse import observe
from nodes.ocr import ocr_node
from nodes.router import route_input
from langgraph.checkpoint.memory import MemorySaver
from nodes.input_guardrail import input_guard_node
from nodes.output_guardrail import output_guard_node
from nodes.reranker import reranker_node
from nodes.merger import merge_node

def build_graph(checkpointer):
    workflow = StateGraph(ChatState)

    workflow.add_node("classifier", classify_node)
    workflow.add_node("topic", topic_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("ocr", ocr_node)
    workflow.add_node("merge", merge_node)
    workflow.add_node("input_guard", input_guard_node)
    workflow.add_node("output_guard", output_guard_node)
    workflow.add_node("reranker", reranker_node)

    # START → input guard
    workflow.add_edge(START, "input_guard")
    workflow.add_conditional_edges(
    "input_guard",
    route_input,
    {
        "block": END,
        "image": "ocr",
        "text": "merge"
    }
)

    workflow.add_edge("ocr", "merge")
    #workflow.add_edge("text","merge")
    workflow.add_edge("merge","classifier")

    workflow.add_conditional_edges(
        "classifier",
        route_query,
        {
            "rag": "topic",
            "chat": "chat"
        }
    )

    workflow.add_edge("topic", "retriever")
    workflow.add_edge("retriever", "reranker")
    workflow.add_edge("reranker", "generator")
    workflow.add_edge("generator", "output_guard")
    workflow.add_edge("chat","output_guard")
    workflow.add_edge("output_guard", END)

    return workflow.compile(checkpointer=checkpointer)


from langfuse import get_client
from langfuse.langchain import CallbackHandler

import os
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-9d2fef24-0400-497a-bbf8-01755b26690f"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-d138e1ac-5a4d-4317-bb4d-6536c8f1017d"
os.environ["LANGFUSE_BASE_URL"] = "https://us.cloud.langfuse.com"
# Initialize Langfuse client
langfuse = get_client()

# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()



def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
