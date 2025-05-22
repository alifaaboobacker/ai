from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
from vector import ingest_markdown, initialize_chroma_db, query_knowledge_base
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

origin = os.getenv("ORIGIN", "*")  # fallback to "*" if ORIGIN is not set
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError("Missing Hugging Face token in environment (HF_TOKEN)")

# FastAPI app
app = FastAPI()

# Initialize ChromaDB collection
chroma_client, collection = initialize_chroma_db()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Request model
class QueryRequest(BaseModel):
    question: str
    section: Optional[str] = None

def call_ollama_llm(context: str, question: str, model: str = "mistralai/Mistral-7B-Instruct-v0.1") -> str:
    prompt = f"""
You are an intelligent and friendly assistant. Based on the following context, answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt}

    response = requests.post(api_url, headers=headers, json=payload)
    try:
        response.raise_for_status()
        output = response.json()
        return output[0]["generated_text"].split("Answer:")[-1].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hugging Face Error: {e}")

# Optional: Trigger vector ingestion manually (saves memory on startup)
@app.post("/load-docs")
def load_docs():
    ingest_markdown("./assets/alifa_knowledgebase.md", collection)
    return {"message": "Documents loaded successfully"}

@app.post("/chat")
def chat(request: QueryRequest):
    results = query_knowledge_base(collection, request.question, section_filter=request.section)
    if not results['documents'][0]:
        return {"question": request.question, "answer": "No relevant context found."}
    context = "\n\n".join(results['documents'][0])
    answer = call_ollama_llm(context, request.question)
    return {"question": request.question, "answer": answer}
