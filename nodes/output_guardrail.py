from langchain_core.messages import AIMessage
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from langfuse import observe
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# reuse abusive list
ABUSIVE_WORDS = [
    "fuck", "shit", "bitch", "idiot", "stupid"
]

def clean_abusive(text: str) -> str:
    for word in ABUSIVE_WORDS:
        text = text.replace(word, "***")
    return text

@observe()
def output_guard_node(state):
    response = state.get("response", "")

    if not response:
        response = "Sorry, I couldn't generate a response."
    response = clean_abusive(response)
    results = analyzer.analyze(
        text=response,
        entities=["EMAIL_ADDRESS", "PHONE_NUMBER"],
        language="en"
    )

    if results:
        response = anonymizer.anonymize(
            text=response,
            analyzer_results=results
        ).text
    return {
        "messages": [AIMessage(content=response)]
    }


