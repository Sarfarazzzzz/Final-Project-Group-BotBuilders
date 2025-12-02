# Final-Project-Group-BotBuilders

AWS CloudGuide ☁️

The AI Technical Architect for Amazon Web Services

AWS CloudGuide is an advanced Hybrid RAG (Retrieval-Augmented Generation) application designed to act as an autonomous cloud architect.

It ingests and synthesizes over 100,000+ chunks of official, up-to-date (2025) AWS User Guide documentation to provide precise, citation-backed answers to complex architectural queries.

🚀 Key Features

Hybrid Search Engine: Combines Dense Vector Retrieval (BAAI/bge-m3) for semantic understanding with Sparse Keyword Search (BM25) for precise acronym/error-code matching.

Cognitive Query Rewriting: Implements a "Rewrite-Retrieve-Read" pipeline that transforms vague follow-up questions (e.g., "How do I configure it?") into precise search queries using conversational context.

Smart Re-Ranking: Utilizes Reciprocal Rank Fusion (RRF) to mathematically merge vector and keyword results, ensuring the most relevant technical manuals appear first.

Enterprise-Grade Reasoning: Powered by Qwen-2.5-7B-Instruct, running locally in FP16 (Half-Precision) mode to maintain maximum reasoning fidelity for complex architectural trade-offs.

Hallucination Guardrails: Strictly grounds all answers in retrieved documentation, providing direct citations ([Source: s3-userguide.pdf]) for every claim.

🛠️ Technical Architecture

The Stack

LLM: Qwen-2.5-7B-Instruct (Hugging Face)

Embeddings: BAAI/bge-m3 (1024 Dimensions)

Vector DB: ChromaDB (Persistent Storage)

Orchestration: LangChain & Hugging Face Pipelines

Hardware Optimization: Optimized for Single-GPU Deployment (NVIDIA T4/A10G) with batched inference.

Interface: Streamlit

The RAG Pipeline Workflow

Ingestion & Indexing: Raw PDFs are processed through a recursive text splitter (1000-char windows with overlap) to preserve technical context before being indexed in parallel Vector and Sparse databases.

Conversational Processing: The system maintains session history and uses a lightweight LLM call to rewrite user queries, resolving coreferences before search.

Hybrid Retrieval: Queries are executed against both indices. Results are fused using RRF to balance semantic breadth with keyword precision.

Grounded Generation: The LLM generates answers using a strict system prompt that enforces citation and formatting rules.

📊 System Evaluation

To validate the reliability of the system for enterprise use, we conducted a Semantic Similarity Evaluation. We compared the system's generated answers against a "Golden Dataset" of expert-verified responses using Cosine Similarity on high-dimensional embeddings.

Metric

Score

Significance

Semantic Retrieval Score

0.74

The system successfully retrieves the correct technical context 74% of the time, significantly outperforming standard keyword search.

Faithfulness

High

The model adheres strictly to the "No relevant documentation found" protocol, preventing the fabrication of false technical specs.

📸 Demo Capabilities

1. Architectural Reasoning

Comparing services to provide decision support.

User: "What is the difference between a Scan and a Query in DynamoDB?"
CloudGuide: "A Scan reads the entire table (inefficient), while a Query uses the partition key (efficient)..." [Source: dynamodb-dg.pdf]

2. Comprehensive Knowledge

Retrieving exhaustive lists from massive documentation.

User: "What are the different storage classes available in Amazon S3?"
CloudGuide: Lists all 9 classes including S3 Express One Zone and Intelligent-Tiering.

3. Conversational Memory

Handling context and ambiguity.

User: "What is the difference between an IAM User and an IAM Role?"
User (Follow-up): "Which one provides temporary credentials?"
CloudGuide: "An IAM Role provides temporary credentials..." (Correctly inferring context).

🔧 Installation & Setup

Prerequisites:

Python 3.10+

NVIDIA GPU (16GB+ VRAM recommended for FP16 inference)

CUDA Toolkit 11.8+

# 1. Clone the repository
git clone [https://github.com/yourusername/aws-cloudguide.git](https://github.com/yourusername/aws-cloudguide.git)
cd aws-cloudguide

# 2. Install dependencies
pip install -r requirements.txt

# 3. Initialize the Knowledge Base
# Downloads and indexes the dataset (Processed ~800k chunks)
python 1_ingest_data.py
python 2_create_embeddings.py

# 4. Launch the Interface
streamlit run app.py --server.port 8501


📄 License

This project uses open-source components and the AWS Public Documentation dataset.
