import langgraph
from services.state import ChatState
from services.llm import llm,qwenn_llm
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

prompt=ChatPromptTemplate.from_messages([
    ('system','You are an helpful assistant'),
    MessagesPlaceholder(variable_name="history"),
    ('user','{query}')
])


def chat_node(state: ChatState):
    query = state["query"]
    history = state.get("messages", [])[:-1]
    final_prompt=prompt.invoke({
        "history":history,
        "query":query
    })

    response = qwenn_llm.invoke(final_prompt)
    return {
        "response": response
    }


