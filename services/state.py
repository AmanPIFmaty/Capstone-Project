from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document


class ChatState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    query_type: str
    topics: List[str]
    docs: List[Document]
    context: str
