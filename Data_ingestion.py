import os
import json
from datasets import load_dataset

# Configuration
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "aws_docs_2025.jsonl")

# Ensure data directory exists
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

            # Write to JSONL (JSON Lines)
            f.write(json.dumps(doc_object) + '\n')
            processed_count += 1

    print(f"Success! Processed {processed_count} high-quality chunks.")
    print(f" Data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    ingest_data()
