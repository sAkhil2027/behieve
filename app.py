import os  
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# =====================================
# Load Environment Variables
# =====================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

# =====================================
# Groq Client
# =====================================
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# =====================================
# Streamlit Page Config
# =====================================
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF RAG Chatbot")

st.markdown(
    """
    Upload a PDF and ask questions about its content.

    This chatbot uses:
    - PDF Text Extraction
    - Text Chunking
    - HuggingFace Embeddings
    - FAISS Vector Database
    - Groq LLM
    """
)

# =====================================
# Session State
# =====================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================
# Sidebar
# =====================================
with st.sidebar:

    st.header("📂 PDF Upload")

    uploaded_pdf = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    st.markdown("---")

    st.info(
        """
        How RAG Works:

        1. Upload PDF
        2. Extract text
        3. Split into chunks
        4. Create embeddings
        5. Store in FAISS
        6. Retrieve relevant chunks
        7. Send context to LLM
        8. Generate answer
        """
    )

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# =====================================
# Process PDF
# =====================================
if uploaded_pdf and st.sidebar.button("Process PDF"):

    try:

        with st.spinner("Processing PDF..."):

            pdf_reader = PdfReader(uploaded_pdf)

            text = ""

            for page in pdf_reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text

            if not text.strip():
                st.error("Could not extract text from PDF.")
                st.stop()

            # Split Text
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_text(text)

            st.success(f"Created {len(chunks)} chunks")

            # Embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Create Vector Store
            vectorstore = FAISS.from_texts(
                texts=chunks,
                embedding=embeddings
            )

            st.session_state.vectorstore = vectorstore

            st.success("✅ PDF processed successfully!")

    except Exception as e:
        st.error(f"Error: {str(e)}")

# =====================================
# Display Chat History
# =====================================
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================
# User Query
# =====================================
query = st.chat_input("Ask a question about your PDF...")

if query:

    if st.session_state.vectorstore is None:
        st.warning("Please upload and process a PDF first.")
        st.stop()

    # -------------------------------
    # User Message
    # -------------------------------
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    # -------------------------------
    # Retrieve Relevant Chunks
    # -------------------------------
    docs = st.session_state.vectorstore.similarity_search(
        query,
        k=4
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Show Retrieved Chunks (Learning Purpose)
    with st.expander("🔍 Retrieved Chunks"):

        for i, doc in enumerate(docs):

            st.write(f"Chunk {i+1}")

            st.write(doc.page_content[:500])

            st.markdown("---")

    # -------------------------------
    # System Prompt
    # -------------------------------
    system_prompt = f"""
You are a helpful PDF Question Answering Assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, say:

"I couldn't find this information in the PDF."

Context:
{context}
"""

    # -------------------------------
    # Generate Response
    # -------------------------------
    with st.chat_message("assistant"):

        placeholder = st.empty()

        full_response = ""

        try:

            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=0,
                stream=True
            )

            for chunk in stream:

                if (
                    chunk.choices
                    and chunk.choices[0].delta.content
                ):

                    full_response += (
                        chunk.choices[0].delta.content
                    )

                    placeholder.markdown(
                        full_response + "▌"
                    )

            placeholder.markdown(full_response)

        except Exception as e:

            full_response = f"Error: {str(e)}"

            placeholder.error(full_response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )