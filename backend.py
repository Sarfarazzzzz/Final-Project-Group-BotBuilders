import os
import pickle
import torch
import warnings
from datetime import datetime
from pathlib import Path
import json
import gc

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

warnings.filterwarnings("ignore")

# Configuration
DB_DIR = "db/chroma_db"
BM25_FILE = "db/bm25_index.pkl"
LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
CONVERSATIONS_DIR = "conversations"

# NEW: Session settings
MAX_HISTORY_TURNS = 10


class RAGBackend:
    def __init__(self):
        self.vector_db = None
        self.bm25 = None
        self.llm_pipeline = None
        self.embedding_model = None
        self.tokenizer = None

        # NEW: Session management
        self.sessions = {}
        Path(CONVERSATIONS_DIR).mkdir(exist_ok=True)
        self.current_session_id = None

        # NEW: Store the raw model for rewriting
        self.model = None

    def load_resources(self):
        print("Loading Resources...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

        if os.path.exists(DB_DIR):
            self.vector_db = Chroma(
                persist_directory=DB_DIR,
                embedding_function=self.embedding_model
            )
            print("Vector DB Loaded.")
        else:
            raise Exception("Vector DB not found!")

        if os.path.exists(BM25_FILE):
            with open(BM25_FILE, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["pipeline"]
                self.bm25_docs = data["documents"]
            print("BM25 Index Loaded.")
        else:
            raise Exception("BM25 Index not found!")

        print(f"Loading LLM: {LLM_MODEL_ID} (Standard 16-Bit)...")

        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # NEW: Store raw model for rewriting step
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        pipe = pipeline(
            "text-generation",
            model=self.model,  # Use self.model
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        self.llm_pipeline = HuggingFacePipeline(pipeline=pipe)
        print("LLM Fully Loaded.")

    # NEW: Session management methods
    def set_session(self, session_id):
        """Set the current session ID - call from Streamlit."""
        self.current_session_id = session_id
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "created": datetime.now().isoformat(),
            }

    # NEW: REWRITE-RETRIEVE-READ - Step 1: REWRITE
    def _rewrite_query(self, query, conversation_history):
        """
        Use LLM to rewrite vague queries into specific search queries.
        This is a separate, lightweight LLM call before retrieval.
        """
        # If no history or query is already specific, skip rewriting
        if not conversation_history or len(query.split()) > 10:
            return query

        # Check if query needs rewriting (has pronouns or is short)
        query_lower = query.lower()
        needs_rewrite = any(word in query_lower.split() for word in ["it", "that", "them", "this", "these", "those"])
        needs_rewrite = needs_rewrite or len(query.split()) <= 6

        if not needs_rewrite:
            return query

        # Build rewriting prompt with conversation history
        rewrite_messages = [
            {
                "role": "system",
                "content": """You are a query rewriting assistant. Your job is to rewrite vague follow-up questions into specific, standalone questions.

Rules:
1. Look at the conversation history to understand context
2. If the user's question has pronouns like "it", "that", "them", replace them with what they refer to
3. Make the question clear and specific
4. Keep the same intent - don't change what the user is asking
5. Output ONLY the rewritten question, nothing else
6. Keep it concise

Examples:
History: "What is Amazon Q?"
Question: "how can I configure it?"
Rewritten: "How can I configure Amazon Q?"

History: "What is AWS Lambda?" | "Lambda is a serverless service"
Question: "what's the timeout?"
Rewritten: "What is the timeout limit for AWS Lambda?"
"""
            }
        ]

        # Use all available history (up to MAX_HISTORY_TURNS) for better context understanding
        if conversation_history:
            recent_history = conversation_history[-(MAX_HISTORY_TURNS * 2):]
            rewrite_messages.extend(recent_history)
            print(f"Using {len(recent_history)} messages for rewriting context")

        # Add the current query to rewrite
        rewrite_messages.append({
            "role": "user",
            "content": f"Rewrite this question to be specific and standalone: {query}"
        })

        # Generate rewritten query using LLM
        try:
            prompt = self.tokenizer.apply_chat_template(
                rewrite_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # Short output
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            rewritten_query = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Clean up
            rewritten_query = rewritten_query.strip('"\'').strip()

            # Sanity check
            if len(rewritten_query.split()) < 3 or len(rewritten_query.split()) > 20:
                print(f" Rewrite failed, using original query")
                return query

            print(f" Query rewritten: '{query}' → '{rewritten_query}'")
            return rewritten_query

        except Exception as e:
            print(f" Rewrite error: {e}, using original query")
            return query

    def reciprocal_rank_fusion(self, results_list, k=60):
        fused_scores = {}
        for doc_list in results_list:
            for rank, doc in enumerate(doc_list):
                doc_key = doc.page_content[:200]
                if doc_key not in fused_scores:
                    fused_scores[doc_key] = {"doc": doc, "score": 0.0}
                fused_scores[doc_key]["score"] += 1.0 / (k + rank)

        reranked_results = sorted(
            fused_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        return [item["doc"] for item in reranked_results]

    def hybrid_search(self, query, top_k=5):
        vector_results = self.vector_db.similarity_search(query, k=top_k * 2)
        tokenized_query = query.lower().split()
        bm25_raw = self.bm25.get_top_n(tokenized_query, self.bm25_docs, n=top_k * 2)
        final_docs = self.reciprocal_rank_fusion([vector_results, bm25_raw])
        return final_docs[:top_k]

    def generate_answer(self, query):
        """
        Generate answer with REWRITE-RETRIEVE-READ and session history.
        SAME SIGNATURE - works with existing app.py!
        """
        # NEW: Get or create session
        if self.current_session_id is None:
            self.current_session_id = "default"
            self.set_session(self.current_session_id)

        session = self.sessions[self.current_session_id]
        conversation_history = session["history"]

        # REWRITE-RETRIEVE-READ: Step 1 - REWRITE
        rewritten_query = self._rewrite_query(query, conversation_history)

        # REWRITE-RETRIEVE-READ: Step 2 - RETRIEVE
        retrieved_docs = self.hybrid_search(rewritten_query)

        context_text = "\n\n".join(
            [f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}"
             for d in retrieved_docs]
        )

        # REWRITE-RETRIEVE-READ: Step 3 - READ
        # NEW: Build messages with conversation history
        messages = [
            {
                "role": "system",
                "content": f"""You are a precise AWS Technical Assistant.

Answer the user's question using ONLY the context provided below.

Formatting Rules (IMPORTANT):
1. **Numbered Steps**: Use this format:
   1. First step here.

   2. Second step here.

   3. Third step here.

2. **Bullet Lists**: Use hyphens with space:
   - Item one
   - Item two

3. **Code/Commands**: Put on separate lines with backticks:
   `aws s3 cp file.txt s3://bucket/`

4. **Paragraphs**: Keep short (2-3 sentences). Add blank line between paragraphs.

5. **Sub-sections**: Use blank lines to separate different topics.

If the context does not contain the answer, say "No relevant documentation found."
Do not say "Based on the context" or "Here is the answer".

Context Data:
{context_text}"""
            }
        ]

        # conversation history
        if conversation_history:
            history_to_include = conversation_history[-(MAX_HISTORY_TURNS * 2):]
            messages.extend(history_to_include)

        # Add current query (use ORIGINAL query, not rewritten)
        messages.append({
            "role": "user",
            "content": query  # Original query!
        })

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Use the raw pipeline to avoid LangChain confusion with chat templates
        outputs = self.llm_pipeline.pipeline(
            prompt,
            max_new_tokens=1024,
            return_full_text=False,
            temperature=0.1,
            top_p=0.9
        )

        response = outputs[0]["generated_text"]

        # Cleanup
        if "<|im_start|>" in response:
            response = response.split("assistant\n")[-1].replace("<|im_end|>", "").strip()

        # NEW: Format the response for better readability
        response = self._format_response(response)

        # NEW: Update conversation history (store clean Q&A)
        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": response})

        return response, retrieved_docs

    def _format_response(self, response):
        """
        Format the response for better readability.

        Improvements:
        - Add line breaks after periods in numbered lists
        - Add spacing after colons and semicolons
        - Clean up markdown code blocks
        - Structure step-by-step instructions
        """
        import re

        # 1. Fix numbered lists (add line breaks after numbers)
        # "1. Step one 2. Step two" → "1. Step one\n\n2. Step two"
        response = re.sub(r'(\d+\.\s[^.]+?\.)(\s+)(\d+\.)', r'\1\n\n\3', response)

        # 2. Add line breaks before numbered items if missing
        # Make sure each "1.", "2.", etc. starts on a new line
        response = re.sub(r'([^\n])(\s+)(\d+\.\s)', r'\1\n\n\3', response)

        # 3. Add spacing after colons in lists
        # "item:description" → "item: description"
        response = re.sub(r':(\S)', r': \1', response)

        # 4. Format code blocks properly
        response = response.replace('```', '\n```\n')

        # 5. Add line breaks before "Additionally," "Furthermore," etc.
        transition_words = ['Additionally', 'Furthermore', 'Moreover', 'Also', 'Note that', 'Important:']
        for word in transition_words:
            response = response.replace(f' {word}', f'\n\n{word}')

        # 6. Clean up extra whitespace
        response = re.sub(r'\n{3,}', '\n\n', response)  # Max 2 newlines
        response = re.sub(r' {2,}', ' ', response)  # Max 1 space

        return response.strip()


if __name__ == "__main__":
    print("🚀 Starting Terminal Test Mode...")
    backend = RAGBackend()
    backend.load_resources()

    # NEW: Create a session for testing
    backend.set_session("test_session")

    # Test conversation flow
    print("\n" + "=" * 70)
    print("  TESTING CONVERSATIONAL RAG")
    print("=" * 70)

    test_questions = [
        "What is Amazon Q?",
        "how can I configure it?",
        "what about pricing?"
    ]

    for i, test_question in enumerate(test_questions, 1):
        print(f"\n{'=' * 70}")
        print(f"Turn {i}: {test_question}")
        print('=' * 70)
        print("Thinking...")

        response, sources = backend.generate_answer(test_question)

        print("\nAI ANSWER:")
        print(response)

        print("\nSOURCES RETRIEVED:")
        for j, doc in enumerate(sources[:2]):
            print(f"[{j + 1}] Source: {doc.metadata.get('source', 'Unknown')}")