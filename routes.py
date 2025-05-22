from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import ollama
from vector import ingest_markdown
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from vector import initialize_chroma_db, query_knowledge_base

load_dotenv()
origin = os.getenv("ORIGIN")
# FastAPI app
app = FastAPI()


# Initialize ChromaDB collection
chroma_client, collection = initialize_chroma_db()
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origin,            # Only allow trusted domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],    # Specify allowed methods
    allow_headers=["*"],              # Or specify allowed headers
)

# Request model
class QueryRequest(BaseModel):
    question: str
    section: Optional[str] = None

def call_ollama_llm(context: str, question: str, model: str = "llama3") -> str:
    prompt = f"""
You are an intelligent and friendly assistant. Based on the following context, answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']
@app.on_event("startup")
def load_vectors():
    ingest_markdown("./assets/alifa_knowledgebase.md", collection)

@app.post("/chat")
def chat(request: QueryRequest):
    results = query_knowledge_base(collection, request.question, section_filter=request.section)
    context = "\n\n".join(results['documents'][0])
    answer = call_ollama_llm(context, request.question)
    return {"question": request.question, "answer": answer}
