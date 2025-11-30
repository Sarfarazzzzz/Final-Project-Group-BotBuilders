import os
import pickle
import torch
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configuration
DB_DIR = "db/chroma_db"
BM25_FILE = "db/bm25_index.pkl"
# Using a powerful 7B model that fits on your A10G
LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


class RAGBackend:
    def __init__(self):
        self.vector_db = None
        self.bm25 = None
        self.llm_pipeline = None
        self.embedding_model = None

    def load_resources(self):
        """Loads DB, Indexes, and LLM. call this once at startup."""
        print(" Loading Resources...")

        # 1. Load Embedding Model (Used for Querying Chroma)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

        # 2. Load Vector DB
        if os.path.exists(DB_DIR):
            self.vector_db = Chroma(
                persist_directory=DB_DIR,
                embedding_function=self.embedding_model
            )
            print(" Vector DB Loaded.")
        else:
            raise Exception("Vector DB not found! Run the embedding script first.")

        # 3. Load BM25 Index
        if os.path.exists(BM25_FILE):
            with open(BM25_FILE, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["pipeline"]
                self.bm25_docs = data["documents"]  # Needed to map index -> text
            print(" BM25 Index Loaded.")
        else:
            raise Exception("BM25 Index not found!")

        # 4. Load LLM (The Generator)
        print(f"Loading LLM: {LLM_MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,  # Half-precision to save VRAM
            trust_remote_code=True
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9
        )
        self.llm_pipeline = HuggingFacePipeline(pipeline=pipe)
        print(" LLM Fully Loaded.")

    def reciprocal_rank_fusion(self, results_list, k=60):
        """
        Combines multiple ranked lists (Vector + Keyword) using RRF algorithm.
        Score = 1 / (k + rank)
        """
        fused_scores = {}

        for doc_list in results_list:
            for rank, doc in enumerate(doc_list):
                # Create a unique key for the document (using content hash or source)
                # Using source+content snippet as key since we didn't save strict IDs
                doc_key = doc.page_content[:200]

                if doc_key not in fused_scores:
                    fused_scores[doc_key] = {"doc": doc, "score": 0.0}

                # RRF Formula
                fused_scores[doc_key]["score"] += 1.0 / (k + rank)

        # Sort by highest score
        reranked_results = sorted(
            fused_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        # Return just the Document objects
        return [item["doc"] for item in reranked_results]

    def hybrid_search(self, query, top_k=5):
        """Performs Vector Search + Keyword Search and fuses results."""

        # 1. Vector Search (Semantic)
        vector_results = self.vector_db.similarity_search(query, k=top_k * 2)

        # 2. BM25 Search (Keyword)
        # Tokenize query
        tokenized_query = query.lower().split()
        # Get top-n raw docs from BM25
        bm25_raw = self.bm25.get_top_n(tokenized_query, self.bm25_docs, n=top_k * 2)

        # 3. Fuse Results (RRF)
        final_docs = self.reciprocal_rank_fusion([vector_results, bm25_raw])

        return final_docs[:top_k]

    def generate_answer(self, query):
        """Full RAG Pipeline: Search -> Prompt -> Generate"""

        # 1. Retrieve
        retrieved_docs = self.hybrid_search(query)

        # 2. Build Context String
        context_text = "\n\n".join(
            [f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}"
             for d in retrieved_docs]
        )

        # 3. Construct Prompt (Chat Format for Qwen)
        prompt = f"""You are a highly technical AWS Cloud Architect Assistant. 
Answer the user's question STRICTLY based on the provided context. 
If the answer is not in the context, say "I cannot find that information in the AWS documentation."

Context:
{context_text}

Question: 
{query}

Answer:"""

        # 4. Generate
        response = self.llm_pipeline.invoke(prompt)

        return response, retrieved_docs


#test block
if __name__ == "__main__":
    print(" Starting Terminal Test Mode...")

    # Initialize the Brain
    backend = RAGBackend()
    backend.load_resources()  # <--- This will take ~30-60 seconds to load the 7B Model

    # Ask a Test Question
    test_question = "How do I create a Lambda function using the console?"
    print(f"\n Question: {test_question}")
    print(" Thinking... (Searching 800k docs + Generating Answer)")

    # Get Answer
    response, sources = backend.generate_answer(test_question)

    print("\n" + "=" * 50)
    print(" AI ANSWER:")
    print("=" * 50)
    print(response)

    print("\n" + "=" * 50)
    print(" SOURCES RETRIEVED:")
    print("=" * 50)
    for i, doc in enumerate(sources):
        print(f"[{i + 1}] Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"    Snippet: {doc.page_content[:150]}...\n")