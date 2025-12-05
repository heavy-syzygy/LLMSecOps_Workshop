from transformers import pipeline

qa_pipeline = pipeline("question-answering",
                       model="distilbert-base-uncased-distilled-squad")

context = "Hugging Face is a technology company that provides open-source NLP libraries ..."
question = "What does Hugging Face provide?"

answer = qa_pipeline(question=question, context=context)
print("Question:", question)
print("Answer:", answer["answer"])
