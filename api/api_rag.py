from fastapi import FastAPI
from pydantic import BaseModel
from scripts.chatbot_rag import PulsEventRAG

app = FastAPI(
    title="PulsEvent RAG API",
    description="API REST pour interroger le système RAG PulsEvent",
    version="1.0"
)

rag = PulsEventRAG()

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(payload: AskRequest):
    if not payload.question.strip():
        return {"error": "La question ne peut pas être vide."}
    answer = rag.ask(payload.question)
    return {"question": payload.question, "answer": answer}

@app.post("/rebuild")
def rebuild_index():
    msg = rag.rebuild_index()
    return {"status": msg}
