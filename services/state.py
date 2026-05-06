from typing import TypedDict, List, Annotated,Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document




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
