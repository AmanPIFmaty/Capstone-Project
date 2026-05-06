import easyocr
import numpy as np
import cv2

reader = easyocr.Reader(['en'], gpu=False)  

def extract_text_from_bytes(image_bytes: bytes) -> str:
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = reader.readtext(image)
        extracted_texts = [res[1] for res in results]
        text = " ".join(extracted_texts)
        text = " ".join(text.split())
        return text
    except Exception as e:
        return f"OCR failed: {str(e)}"
