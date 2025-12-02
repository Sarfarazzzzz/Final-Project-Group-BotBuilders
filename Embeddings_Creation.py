import json
import os
import pickle
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi

# Configuration
INPUT_FILE = "data/aws_docs_2025.jsonl"
DB_DIR = "db/chroma_db"
BM25_FILE = "db/bm25_index.pkl"


def create_embeddings():
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Embedding Pipeline on device: {device.upper()}")

    if device == "cpu":
        print("WARNING: Running on CPU. This will be slow. Use a GPU instance if possible.")

    # 1. Load the Cleaned Data
    print("PAGE 1: Loading Data...")
    docs = []
    # Check if file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found. Did you run the ingestion script?")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Create a LangChain 'Document' object
            doc = Document(
                page_content=data['text'],
                metadata=data['metadata']
            )
            docs.append(doc)

    print(f"Loaded {len(docs)} raw pages.")

    # 2. Split Large Pages into Precision Chunks
    # We split by 'headers' and 'paragraphs' to keep context intact
    print("PAGE 2: Splitting text into precision chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    splitted_docs = text_splitter.split_documents(docs)
    print(f"Generated {len(splitted_docs)} searchable chunks.")

    # 3. Create Vector Store (The Semantic Index)
    print("PAGE 3: Generating Vectors (This uses the GPU)...")
    print("   -> Loading BAAI/bge-m3 model (this may take a moment)...")

    # We use BGE-M3, a state-of-the-art model for retrieval
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    # This process will take 5-15 mins depending on dataset size
    vector_db = Chroma.from_documents(
        documents=splitted_docs,
        embedding=embedding_model,
        persist_directory=DB_DIR
    )
    print(f"Vector Database Persisted to {DB_DIR}")

    # 4. Create BM25 Index (The Keyword Index)
    print("PAGE 4: Building BM25 Keyword Index...")

    # Simple tokenization for BM25
    tokenized_corpus = [doc.page_content.lower().split() for doc in splitted_docs]
    bm25 = BM25Okapi(tokenized_corpus)

    # Save the BM25 index AND the raw docs
    # (We need the raw docs to map the BM25 'index number' back to text later)
    bm25_data = {
        "pipeline": bm25,
        "documents": splitted_docs
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(BM25_FILE), exist_ok=True)

    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25_data, f)

    print(f"BM25 Index Saved to {BM25_FILE}")

if __name__ == "__main__":
    create_embeddings()
