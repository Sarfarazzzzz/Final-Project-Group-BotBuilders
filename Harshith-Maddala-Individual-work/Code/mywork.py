#%% Evaluate_model.py:
import torch
import warnings
import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from backend import RAGBackend

warnings.filterwarnings("ignore")

# 1. Setup
print(" Loading Evaluator...")
judge_model = SentenceTransformer("BAAI/bge-m3")
rag_system = RAGBackend()
rag_system.load_resources()

# 2. Golden Dataset
test_questions = [
    # Q1: Compute (Lambda) - Fact
    "What is the maximum execution time for an AWS Lambda function?",

    # Q2: Database (DynamoDB) - Concept
    "What is the difference between a Scan and a Query in DynamoDB?",

    # Q3: Storage (S3) - Procedure
    "How do I create an S3 bucket?",

    # Q4: Compute (EC2) - Fact (Tests your EC2 Ghost Fix)
    "What is the difference between On-Demand and Spot Instances?",

    # Q5: Security (IAM) - Concept
    "What is the AWS root user?"
]

# The "Answer Key"
ground_truths = [
    # A1
    "The maximum execution time for an AWS Lambda function is 15 minutes or 900 seconds.",

    # A2
    "A Query finds items based on primary key values and is efficient. A Scan reads every item in the table and is slower and consumes more throughput.",

    # A3
    "To create an S3 bucket, use the AWS Console, click Create Bucket, choose a unique name, and select a region.",

    # A4
    "On-Demand instances are fixed rate by the second with no commitment. Spot instances use spare EC2 capacity for up to 90% discounts but can be interrupted.",

    # A5
    "The root user is the identity created when you first create your AWS account. It has complete, unrestricted access to all resources and billing."
]

print("\n Running Semantic Evaluation...")
scores = []

for i, q in enumerate(test_questions):
    print(f"   Asking: {q}")
    response, source_docs = rag_system.generate_answer(q)

    # Combine all retrieved text into one big string context
    retrieved_text = " ".join([doc.page_content for doc in source_docs])

    # 1. Vectorize the Ground Truth (The Ideal Answer)
    truth_vec = judge_model.encode([ground_truths[i]])

    # 2. Vectorize the Retrieved Context (What your DB found)
    context_vec = judge_model.encode([retrieved_text])

    # 3. Calculate Similarity (0 to 1)
    similarity_score = cosine_similarity(truth_vec, context_vec)[0][0]
    scores.append(similarity_score)

    status = "PASS" if similarity_score > 0.55 else "FAIL"
    print(f"   -> Semantic Score: {similarity_score:.4f} | {status}")
    print(f"   -> Model Answer: {response[:100]}...")
    print("-" * 50)

avg_score = np.mean(scores)
print("\n" + "=" * 40)
print(f"FINAL SEMANTIC SCORE")
print("=" * 40)
print(f"Average Similarity: {avg_score:.4f} (Target: >0.60)")
if avg_score > 0.6:
    print("RESULT: SYSTEM PASSED")
else:
    print("RESULT: SYSTEM NEEDS TUNING")
print("=" * 40)


#%% app.py 
import streamlit as st
import time
import uuid
from backend1 import RAGBackend

# Page Config
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

# Title
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

# ========================================
# Session management
# ========================================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

backend.set_session(st.session_state.session_id)
# ========================================

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
                # Get response
                response_text, source_docs = backend.generate_answer(prompt)

                # Smart Streaming - preserves formatting!
                full_response = ""

                # Split by double newlines (paragraphs) to preserve structure
                paragraphs = response_text.split('\n\n')

                for para_idx, paragraph in enumerate(paragraphs):
                    # Stream words in this paragraph
                    words = paragraph.split()
                    for word in words:
                        full_response += word + " "
                        time.sleep(0.01)  # Faster streaming
                        message_placeholder.markdown(full_response + "▌")

                    # Add paragraph break (except after last paragraph)
                    if para_idx < len(paragraphs) - 1:
                        full_response += "\n\n"
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")

                # Show final version without cursor
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