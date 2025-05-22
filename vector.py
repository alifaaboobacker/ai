from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Tuple, Optional
import os
import uuid
import re

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def initialize_chroma_db(persist_dir: str = "./chroma_db", collection_name: str = "knowledge_base"):
    client = chromadb.Client(Settings(
        persist_directory=persist_dir,
        anonymized_telemetry=False
    ))
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef
    )
    return client, collection

def smart_chunk_markdown(filepath: str) -> List[Tuple[str, str]]:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    sections = re.split(r"\n(?=## )", content)
    chunks = []
    for section in sections:
        if not section.strip():
            continue

        heading_match = re.match(r"##+ (.+)", section)
        heading = heading_match.group(1).strip() if heading_match else "General"

        sub_chunks = re.split(r"\n(?=\*\*Q: )", section)
        for chunk in sub_chunks:
            if chunk.strip():
                chunks.append((chunk.strip(), heading))

    return chunks

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    return model.encode(texts).tolist()

def store_embeddings(collection, texts: List[str], embeddings: List[List[float]], metadatas: List[dict]):
    ids = [str(uuid.uuid4()) for _ in texts]
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

def ingest_markdown(filepath: str, collection):
    chunks = smart_chunk_markdown(filepath)
    if not chunks:
        print("❌ No valid documents found.")
        return

    texts, metadatas = [], []
    for text, section in chunks:
        texts.append(text)
        metadatas.append({"source": os.path.basename(filepath), "section": section})

    embeddings = generate_embeddings(texts)
    store_embeddings(collection, texts, embeddings, metadatas)
    print("✅ Document ingested and stored in ChromaDB.")

def query_knowledge_base(collection, question: str, section_filter: Optional[str] = None, n_results: int = 5):
    query_args = {
        "query_texts": [question],
        "n_results": n_results,
        "where": {"source": "alifa_knowledgebase.md"}
    }
    if section_filter:
        query_args["where"]["section"] = section_filter

    results = collection.query(**query_args)
    return results

def display_results(results):
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print(f"\n📄 Section: {meta.get('section', 'Unknown')}\n📝 Text: {doc}")

if __name__ == "__main__":
    chroma_client, collection = initialize_chroma_db()
    ingest_markdown("./assets/alifa_knowledgebase.md", collection)

    # Example queries
    result = query_knowledge_base(collection, "What is Alifa's work experience?")
    display_results(result)

    result = query_knowledge_base(collection, "What skills does she have?")
    display_results(result)
