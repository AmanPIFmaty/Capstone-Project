def classify_query(query, llm):
    prompt = f"""
You are a query classifier.

Return:
- "rag" → for technical / IT / troubleshooting queries
- "chat" → for casual conversation

Return ONLY one word.

Query: {query}
"""
    response = llm.invoke(prompt).strip().lower()

    if "rag" in response:
        return "rag"
    return "chat"
