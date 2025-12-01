import streamlit as st
import time
from backend import RAGBackend

# Page Config
st.set_page_config(
    page_title="AWS Architect Assistant",
    page_icon="☁️",
    layout="wide"
)

# Custom CSS for a professional look
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
    # Use HTML to render the image directly in the browser (bypassing the server backend)
    st.markdown('<img src="https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg" width="80">', unsafe_allow_html=True)
with col2:
    st.title("AWS Technical Architect Agent")
    st.markdown("*Powered by Hybrid RAG (Dense Vectors + BM25) & Qwen-7B*")


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
        full_response = ""

        with st.spinner("Thinking..."):
            try:
                response_text, source_docs = backend.generate_answer(prompt)

                # Streaming Effect
                for chunk in response_text.split():
                    full_response += chunk + " "
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

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
                    "content": full_response,
                    "sources": clean_sources
                })

            except Exception as e:
                st.error(f"Error: {e}")
