# Data Ingestion

import os
import json
from datasets import load_dataset

# Configuration

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "aws_docs_2025.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def ingest_data():
    print("Starting Data Ingestion Pipeline...")

    # 1. Download the Dataset

    print("Downloading `semihk1/aws-public-pdf-chunked-dataset`...")
    try:
        dataset = load_dataset("semihk1/aws-public-pdf-chunked-dataset", split="train")
        print(f"Download Complete. Total Raw Chunks: {len(dataset)}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    # 2. Process and Format

    print("⚙️ Formatting data for RAG pipeline...")

    processed_count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in dataset:

            text_content = entry.get('text', '').strip()
            source = entry.get('source', 'unknown_source')
            chunk_id = entry.get('chunk_id', -1)



            # 1. Skip Table of Contents
            if "Table of Contents" in text_content or text_content.count(".....") > 3:
                print(f"Skipping TOC: {chunk_id}")
                continue

            # 2. Skip Copyright/Legal boilerplate
            if "All rights reserved" in text_content and len(text_content) < 300:
                print(f"Skipping Copyright: {chunk_id}")
                continue

            # 3. Skip very short chunks
            if len(text_content) < 100:
                continue

            # Create a clean JSON object
            doc_object = {
                "id": f"{source}_{chunk_id}",
                "text": text_content,
                "metadata": {
                    "source": source,
                    "length": len(text_content)
                }
            }

            f.write(json.dumps(doc_object) + '\n')
            processed_count += 1

    print(f"Success! Processed {processed_count} high-quality chunks.")
    print(f" Data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    ingest_data()

#################################################################################################

# Embeddings Creation

import json
import os
import pickle
import torch
import warnings
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi

warnings.filterwarnings("ignore")

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
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found. Did you run the ingestion script?")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)

            doc = Document(
                page_content=data['text'],
                metadata=data['metadata']
            )
            docs.append(doc)

    print(f"Loaded {len(docs)} raw pages.")

    # 2. Split Large Pages into Precision Chunks

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

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_db = Chroma.from_documents(
        documents=splitted_docs,
        embedding=embedding_model,
        persist_directory=DB_DIR
    )
    print(f"Vector Database Persisted to {DB_DIR}")

    # 4. Create BM25 Index (The Keyword Index)
    print("PAGE 4: Building BM25 Keyword Index...")

    tokenized_corpus = [doc.page_content.lower().split() for doc in splitted_docs]
    bm25 = BM25Okapi(tokenized_corpus)

    # Save the BM25 index AND the raw docs
    bm25_data = {
        "pipeline": bm25,
        "documents": splitted_docs
    }

    os.makedirs(os.path.dirname(BM25_FILE), exist_ok=True)

    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25_data, f)

    print(f"BM25 Index Saved to {BM25_FILE}")

if __name__ == "__main__":
    create_embeddings()

############################################################################################

# App

import streamlit as st
import time
import uuid
from backend import RAGBackend

# Page Configuration

st.set_page_config(
    page_title="AWS CloudGuide",
    page_icon="☁️",
    layout="wide"
)

# Custom CSS

st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e; color: #ffffff
    }
    .chat-message.bot {
        background-color: #f0f2f6; color: #000000
    }
    .source-box {
        font-size: 0.9em; color: #444; border-left: 3px solid #ff9900; 
        padding-left: 10px; margin-top: 5px; background-color: #f9f9f9; padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 6])
with col1:
    st.markdown('<img src="https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg" width="80">', unsafe_allow_html=True)
with col2:
    st.title("AWS CloudGuide")
    st.markdown("**Your Cloud Architecture Assistant**")


# Initialize Backend (Cached)

@st.cache_resource
def get_backend():
    backend = RAGBackend()
    backend.load_resources()
    return backend


# Load the brain

with st.spinner("🚀 Booting up GPU Engine & Loading Knowledge Base..."):
    try:
        backend = get_backend()
        st.success("System Online!")
    except Exception as e:
        st.error(f"Error Loading System: {e}")
        st.stop()

# Session management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

backend.set_session(st.session_state.session_id)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 View Sources"):
                for src in message["sources"]:
                    st.markdown(f"<div class='source-box'><b>📄 {src['source']}</b><br><i>{src['content']}</i></div>",
                                unsafe_allow_html=True)

# User Input
if prompt := st.chat_input("Ask a technical question (e.g., 'How do I create an S3 bucket?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.spinner("Thinking..."):
            try:

                response_text, source_docs = backend.generate_answer(prompt)

                full_response = ""

                # Split by double newlines (paragraphs) to preserve structure
                paragraphs = response_text.split('\n\n')

                for para_idx, paragraph in enumerate(paragraphs):
                    words = paragraph.split()
                    for word in words:
                        full_response += word + " "
                        time.sleep(0.01)
                        message_placeholder.markdown(full_response + "▌")

                    # Add paragraph break
                    if para_idx < len(paragraphs) - 1:
                        full_response += "\n\n"
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(response_text)

                # Format Sources
                clean_sources = []
                for doc in source_docs[:3]:
                    clean_sources.append({
                        "source": doc.metadata.get("source", "AWS Docs"),
                        "content": doc.page_content[:300] + "..."
                    })

                with st.expander("📚 View Sources"):
                    for src in clean_sources:
                        st.markdown(
                            f"<div class='source-box'><b>📄 {src['source']}</b><br><i>{src['content']}</i></div>",
                            unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": clean_sources
                })

            except Exception as e:
                st.error(f"Error: {e}")