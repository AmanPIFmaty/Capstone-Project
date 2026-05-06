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
    image_bytes: bytes

# %%
from nodes.classifier import classify_node
from nodes.topic import topic_node
from nodes.retriever import retriever_node
from nodes.generator import generator_node
from nodes.chat import chat_node
from nodes.router import route_query
from langfuse import observe
from nodes.ocr import ocr_node
from nodes.router import route_input
from langgraph.checkpoint.memory import MemorySaver
from nodes.input_guardrail import input_guard_node
from nodes.output_guardrail import output_guard_node
from nodes.reranker import reranker_node

def build_graph():
    workflow = StateGraph(ChatState)

    workflow.add_node("classifier", classify_node)
    workflow.add_node("topic", topic_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("ocr", ocr_node)
    workflow.add_node("input_guard", input_guard_node)
    workflow.add_node("output_guard", output_guard_node)
    workflow.add_node("reranker", reranker_node)

    # START → input guard
    workflow.add_edge(START, "input_guard")
    workflow.add_conditional_edges(
    "input_guard",
    route_input,
    {
        "block": END,       # ✅ abusive → stop immediately, no LLM wasted
        "image": "ocr",
        "text": "classifier"
    }
)

    workflow.add_edge("ocr", "classifier")

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
    workflow.add_edge("chat", "output_guard")
    workflow.add_edge("output_guard", END)
    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


# %%
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

# %%
from langchain_core.messages import HumanMessage

chatbot = build_graph()
config = {"configurable": {"thread_id": "th-1"},"callbacks":[langfuse_handler]}

while True:
    print("\nChoose input type:")
    print("1. Text")
    print("2. Image")
    print("Type 'quit' to exit")

    choice = input("Enter choice: ").strip()

    if choice.lower() == "quit":
        break

    if choice == "1":
        query = input("Enter your query: ")
        print("User: ",query)
        response = chatbot.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )

    elif choice == "2":
        image_path = input("Enter image file path: ").strip()

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            print("System: Image uploaded")
            response = chatbot.invoke(
                {"image_bytes": image_bytes},
                config=config
            )

        except Exception as e:
            print("System: Error loading image:", e)
            continue

    else:
        print("System: Invalid choice. Try again.")
        continue

    print("\nAI:", response["messages"][-1].content)

# %%
from IPython.display import Image, display

# Assuming 'app' is your compiled LangGraph workflow
display(Image(chatbot.get_graph().draw_mermaid_png()))

# %%
from langchain_core.messages import HumanMessage

chatbot = build_graph()
config = {
    "configurable": {"thread_id": "th-1"},
    "callbacks": [langfuse_handler]
}

while True:
    print("\nChoose input type:")
    print("1. Text")
    print("2. Image")
    print("Type 'quit' to exit")

    choice = input("Enter choice: ").strip()

    if choice.lower() == "quit":
        break

    if choice == "1":
        query = input("Enter your query: ").strip()
        print("User: ", query)

        response = chatbot.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "query": query,      
                "image_bytes": None  
            },
            config=config
        )

    elif choice == "2":
        image_path = input("Enter image file path: ").strip()

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            print("System: Image uploaded")

            response = chatbot.invoke(
                {
                    "image_bytes": image_bytes,
                    "query": "",     
                    "messages": []   
                },
                config=config
            )

        except FileNotFoundError:
            print(f"System: File not found: {image_path}")
            continue
        except Exception as e:
            print(f"System: Error loading image: {e}")
            continue

    else:
        print("System: Invalid choice. Try again.")
        continue

    print("\nAI:", response["messages"][-1].content)

# %%
