from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import requests
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

def call_ollama_llm(context: str, question: str, model: str = "mistralai/Mistral-7B-Instruct-v0.1") -> str:
    prompt = f"""
You are an intelligent and friendly assistant. Based on the following context, answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
    hf_token = os.getenv("HF_TOKEN")
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt}

    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()

    output = response.json()
    return output[0]["generated_text"].split("Answer:")[-1].strip()
@app.on_event("startup")
def load_vectors():
    ingest_markdown("./assets/alifa_knowledgebase.md", collection)

@app.post("/chat")
def chat(request: QueryRequest):
    results = query_knowledge_base(collection, request.question, section_filter=request.section)
    context = "\n\n".join(results['documents'][0])
    answer = call_ollama_llm(context, request.question)
    return {"question": request.question, "answer": answer}
