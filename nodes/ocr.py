from langchain_core.messages import HumanMessage
from langfuse import observe
from services.ocr import extract_text_from_bytes

@observe()
def ocr_node(state):
    image_bytes = state["image_bytes"]
    text = extract_text_from_bytes(image_bytes)

    return {
        #"messages": [HumanMessage(content=text)], 
        "ocr_text": text,
        "image_bytes": None
    }

