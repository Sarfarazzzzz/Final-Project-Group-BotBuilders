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