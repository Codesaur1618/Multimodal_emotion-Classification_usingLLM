from transformers import pipeline

text_classifier = pipeline("text-classification", model="ayoubkirouane/BERT-Emotions-Classifier")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def extract_emotion_cause(context):
    query = "What caused the emotion?"
    try:
        result = qa_pipeline(question=query, context=context)
        return result["answer"]
    except Exception as e:
        return f"Error extracting cause: {str(e)}"

def classify_text_emotion(texts):
    results = []
    for text in texts:
        try:
            result = text_classifier(text, truncation=True, max_length=512)
            results.append({
                "utterance": text,
                "emotion": result[0]["label"],
                "confidence": result[0]["score"]
            })
        except Exception as e:
            results.append({"utterance": text, "emotion": "error", "confidence": 0.0, "error": str(e)})
    return results
