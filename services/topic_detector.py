import json
from services.vectorstore import all_chunks

topic_list = []
for chunk in all_chunks:
    topic = chunk.metadata.get("topic") or chunk.metadata.get("metadata_topic")
    if topic and topic not in topic_list:
        topic_list.append(topic)


def detect_topic_from_query(query, topic_list, llm):
    prompt = f"""
You are a topic classifier. Given a user query and a list of available topics, 
return ONLY the most relevant topic names from the list as a JSON array.
Return maximum 3 topics. Return empty array [] if nothing matches.
Do not explain.

Available Topics:
{json.dumps(topic_list)}

User Query: {query}

Response (JSON array only):
"""
    response = llm.invoke(prompt).strip()

    try:
        topics = json.loads(response)
        return [t for t in topics if t in topic_list]
    except:
        return []
