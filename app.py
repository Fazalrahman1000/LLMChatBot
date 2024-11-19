import os
import json
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from groq import Groq

# Constants
UPLOADS_FOLDER = "uploads"
CHUNK_SIZE = 2048
OVERLAP = 512
st.set_page_config(page_title="Document Q&A Bot", page_icon="📄")

# Load API key and admin password from config.json
config_file = "config.json"
if os.path.exists(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
else:
    st.error("Config file not found. Ensure config.json is present.")
    st.stop()

# Set the Groq API key and admin password
GROQ_API_KEY = config.get("GROQ_API_KEY")
ADMIN_PASSWORD = config.get("ADMIN_PASSWORD")

if not GROQ_API_KEY or not ADMIN_PASSWORD:
    st.error("GROQ_API_KEY or ADMIN_PASSWORD missing in config.json.")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize the Groq client
client = Groq(api_key=GROQ_API_KEY)

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_size = embedding_model.get_sentence_embedding_dimension()

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = faiss.IndexFlatL2(embedding_size)
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings_added" not in st.session_state:
    st.session_state.embeddings_added = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Ensure uploads folder exists
os.makedirs(UPLOADS_FOLDER, exist_ok=True)

# Authentication
def login():
    st.session_state.authenticated = True

def logout():
    st.session_state.authenticated = False

# Sidebar for login
with st.sidebar:
    if not st.session_state.authenticated:
        st.header("Login")
        password = st.text_input("Enter the password", type="password")
        if st.button("Login"):
            if password == ADMIN_PASSWORD:
                login()
                st.success("Logged in successfully!")
            else:
                st.error("Invalid password.")
    else:
        st.button("Logout", on_click=logout)
        st.success("You are logged in!")

# Function to load documents
def load_documents(directory):
    """Load text content from all documents in the directory."""
    documents = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.endswith('.pdf'):
            try:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                documents.append(text)
            except Exception as e:
                st.error(f"Error reading {file_name}: {e}")
        elif file_name.endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    documents.append(f.read())
            except Exception as e:
                st.error(f"Error reading {file_name}: {e}")
    return documents

# Function to process and chunk documents
def process_and_chunk_documents(directory):
    """Process documents in the directory and create text chunks with overlap."""
    documents = load_documents(directory)
    chunks = []
    for doc in documents:
        for i in range(0, len(doc), CHUNK_SIZE - OVERLAP):
            chunks.append(doc[i:i + CHUNK_SIZE])
    return chunks

# Function to generate embeddings for chunks and add to FAISS index
def add_chunks_to_faiss_index(chunks):
    """Generate embeddings for text chunks and add to FAISS index."""
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True).astype('float32')
    st.session_state.faiss_index.add(embeddings)
    st.session_state.chunks = chunks
    st.session_state.embeddings_added = True

# Process documents in the uploads folder
if not st.session_state.embeddings_added:
    chunks = process_and_chunk_documents(UPLOADS_FOLDER)
    add_chunks_to_faiss_index(chunks)

# Main chatbot interface
if st.session_state.authenticated:
    st.title("📄 Document Q&A Bot")
    st.write("Ask any question about the documents in the `uploads` folder!")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask a question...")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # Search for relevant chunks in FAISS index
        if st.session_state.embeddings_added and st.session_state.chunks:
            try:
                query_embedding = embedding_model.encode([user_prompt], convert_to_numpy=True).astype('float32')
                distances, indices = st.session_state.faiss_index.search(query_embedding, k=5)
                retrieved_chunks = [st.session_state.chunks[idx] for idx in indices[0] if idx < len(st.session_state.chunks)]
                context_text = "\n\n".join(retrieved_chunks)
            except Exception as e:
                st.error(f"Error retrieving context: {e}")
                context_text = ""

        # Send query with context to LLAMA model
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "system", "content": f"Relevant excerpts from documents:\n{context_text}"},
                    {"role": "user", "content": user_prompt}
                ]
            )
            assistant_response = response.choices[0].message.content
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

            with st.chat_message("assistant"):
                st.markdown(assistant_response)

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
else:
    st.info("Please log in to access the chatbot.")
