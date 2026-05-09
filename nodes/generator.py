# generator_node.py
from services.state import ChatState
from services.llm import llm,qwenn_llm
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langfuse import observe

MAX_HISTORY_MESSAGES = 4  # keep last 2 exchanges (user + assistant × 2)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful IT support assistant for troubleshooting and SOPs.
Answer based ONLY on the provided context. If the answer is not in the context, say "I don't have enough information to answer this."
Answer in clear steps or bullet points. Be concise.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

@observe()
def generator_node(state: ChatState):
    query = state.get("final_query")
    context = state.get("context", "")   # use context already built by reranker
    
    all_history = state.get("messages", [])[:-1]
    history = all_history[-MAX_HISTORY_MESSAGES:] if all_history else []

    final_prompt = prompt.invoke({
        "context": context,
        "question": query,
        "history": history
    })

    response = qwenn_llm.invoke(final_prompt)

    return {
        "response": response
    }
