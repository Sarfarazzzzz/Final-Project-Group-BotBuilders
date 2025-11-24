import os
import json
from datasets import load_dataset

# Configuration
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "aws_docs_2025.jsonl")

# Ensure data directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def ingest_data():
    print("🚀 Starting Data Ingestion Pipeline...")

    # 1. Download the Dataset
    # We use 'semihk1' because it contains 2025 AWS User Guide PDFs (High Quality)
    print("📥 Downloading `semihk1/aws-public-pdf-chunked-dataset`...")
    try:
        dataset = load_dataset("semihk1/aws-public-pdf-chunked-dataset", split="train")
        print(f"✅ Download Complete. Total Raw Chunks: {len(dataset)}")
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return

    # 2. Process and Format
    # We need to standardize the format for our Vector DB (Chroma)
    print("⚙️ Formatting data for RAG pipeline...")

    processed_count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in dataset:
            # Extract fields (handling potential missing keys safely)
            text_content = entry.get('text', '').strip()
            source = entry.get('source', 'unknown_source')
            chunk_id = entry.get('chunk_id', -1)

            # Quality Check: Skip empty or extremely short chunks (noise)
            if len(text_content) < 50:
                continue

            # Create a clean JSON object
            doc_object = {
                "id": f"{source}_{chunk_id}",  # Unique ID for the database
                "text": text_content,
                "metadata": {
                    "source": source,
                    "length": len(text_content)
                }
            }

            # Write to JSONL (JSON Lines) - Best for streaming large data
            f.write(json.dumps(doc_object) + '\n')
            processed_count += 1

    print(f"🎉 Success! Processed {processed_count} high-quality chunks.")
    print(f"📂 Data saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    ingest_data()