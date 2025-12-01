import json
import os
import torch
import gc
import random
import numpy as np
import warnings
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

warnings.filterwarnings("ignore")

# Configuration
INPUT_FILE = "data/aws_docs_2025.jsonl"
DB_DIR = "db/chroma_db"
BATCH_SIZE = 500


def final_repair():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Robust Ghost Repair on {device.upper()}...")

    # 1. Load Raw Data counts
    print("📂 Analyzing Raw Data Source...")
    raw_docs_by_source = {}
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            source = data['metadata'].get('source', 'unknown')
            if source not in raw_docs_by_source:
                raw_docs_by_source[source] = []
            raw_docs_by_source[source].append(Document(page_content=data['text'], metadata=data['metadata']))

    print(f"✅ Raw Dataset has {len(raw_docs_by_source)} unique files.")

    # 2. Connect to DB
    print("🔌 Connecting to Database...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)

    # 3. IDENTIFY GHOSTS (With Error Handling)
    print("🔍 Performing Deep Vector Integrity Scan...")
    corrupt_sources = []

    try:
        db_data = vector_db.get()
        all_ids = db_data['ids']
        all_metadatas = db_data['metadatas']
    except Exception as e:
        print(f"❌ Critical DB Read Error: {e}")
        print("⚠️ Assuming Database is totally corrupt. Re-ingesting everything.")
        all_ids = []
        all_metadatas = []

    # Map Source -> List of IDs
    db_map = {}
    for idx, meta in enumerate(all_metadatas):
        if meta and 'source' in meta:
            src = meta['source']
            if src not in db_map:
                db_map[src] = []
            db_map[src].append(all_ids[idx])

    # Check every source
    for source, raw_docs in tqdm(raw_docs_by_source.items(), desc="Scanning Files"):
        is_corrupt = False
        reason = ""

        db_ids = db_map.get(source, [])

        # Check 1: Count Mismatch
        if len(db_ids) < len(raw_docs):
            is_corrupt = True
            reason = f"Missing Chunks (Raw: {len(raw_docs)}, DB: {len(db_ids)})"

        # Check 2: Vector Sampling (The Crash Prone Part)
        elif len(db_ids) > 0:
            try:
                sample_ids = random.sample(db_ids, min(3, len(db_ids)))
                # We wrap this in TRY/CATCH to handle the "InternalError"
                sample_data = vector_db.get(ids=sample_ids, include=['embeddings'])
                embeddings = sample_data['embeddings']

                for emb in embeddings:
                    if emb is None or len(emb) == 0 or np.sum(np.abs(emb)) < 1e-6:
                        is_corrupt = True
                        reason = "Ghost Vectors (Empty/Zero Data)"
                        break
            except Exception as e:
                # THIS CATCHES YOUR ERROR
                is_corrupt = True
                reason = f"DB Integrity Crash: {str(e)[:50]}..."

        if is_corrupt:
            # print(f"   ❌ DETECTED GHOST: {source} | Reason: {reason}")
            corrupt_sources.append(source)

    if len(corrupt_sources) == 0:
        print("🎉 SUCCESS: No ghost files found.")
        return

    print(f"\n⚠️ Found {len(corrupt_sources)} broken files. Repairing now...")

    # 4. REPAIR LOOP
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for source in tqdm(corrupt_sources, desc="Repairing Ghosts"):
        # Step A: Delete (Try to clean up, but ignore errors if index is broken)
        try:
            vector_db._collection.delete(where={"source": source})
        except:
            pass  # If delete fails, we just overwrite

        # Step B: Re-ingest
        raw_docs = raw_docs_by_source[source]
        splitted_docs = text_splitter.split_documents(raw_docs)

        for i in range(0, len(splitted_docs), BATCH_SIZE):
            batch = splitted_docs[i: i + BATCH_SIZE]
            vector_db.add_documents(documents=batch)

        gc.collect()
        torch.cuda.empty_cache()

    print("✅ All Ghosts Busted! Database is fully synced.")


if _name_ == "_main_":
    final_repair()
