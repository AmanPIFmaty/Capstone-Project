from services.state import ChatState


def merge_node(state: ChatState):
    user_text = state.get("query", "")
    ocr_text = state.get("ocr_text", "")

    if user_text and ocr_text:
        final_query = (
            f"User Query:\n{user_text}\n\n"
            f"Image Context:\n{ocr_text}"
        )
    elif ocr_text:
        final_query = ocr_text
    else:
        final_query = user_text

    return {"final_query": final_query}

