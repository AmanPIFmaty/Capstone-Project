def classify_query(query, llm):
    prompt = f"""
You are a strict query classifier.

Rules:
- Return ONLY "rag" for technical / IT / troubleshooting queries
- Return ONLY "chat" for casual conversation like "hi", "hello", "greetings" or other casual small talks.
- Retrun ONLY "chat" if asked about past conversations.
- Return ONLY one word.
- RETURN "rag" if there is nothing in the query
- Do NOT return anything else

Query: {query}
"""

    response = llm.invoke(prompt).strip().lower()

    if response in ["rag", "technical", "tech", "it", "troubleshooting"]:
        return "rag"
    elif response in ["chat", "general", "conversation"]:
        return "chat"

    return "chat"
