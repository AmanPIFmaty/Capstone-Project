import langgraph
from services.state import ChatState
from services.llm import llm
from services.formatting import format_docs
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langfuse import observe

prompt = ChatPromptTemplate.from_template("""
You are a helpful IT support assistant made for troubleshooting and SOPs providing. 
Answer the user's question based only on the provided context.
If the answer is not in the context, say "I don't have enough information to answer this."
Answer in step wise or point wise.

Context:
{context}

Question:
{question}

Answer:
""")

@observe()
def generator_node(state: ChatState):
    query = state["query"]
    docs = state.get("docs", [])

    context = format_docs(docs)


    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)
        

    return {
        "context": context,
        "messages": [AIMessage(content=response)]
    }
