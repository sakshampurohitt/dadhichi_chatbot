import streamlit as st
import os
import tempfile
from io import BytesIO
from types import SimpleNamespace
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Set page configuration
st.set_page_config(page_title="Dadhichi - AI Fitness Coach", page_icon="🧠", layout="wide")

# Custom styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        text-align: center;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
        background-color: #f9f9f9;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .user-message {
        background-color: #e6f3ff;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .assistant-message {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .error-message {
        background-color: #ffebee;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🧠 Dadhichi</h1>', unsafe_allow_html=True)
st.write("I'm your personal AI fitness and nutrition coach powered by Llama 3 Instruct. "
         "Upload fitness-related PDFs to enhance my knowledge!👋")

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

if "uploaded_files_info" not in st.session_state:
    st.session_state.uploaded_files_info = []
    
if "ollama_api_base" not in st.session_state:
    st.session_state.ollama_api_base = "http://localhost:11434"

# Function to process PDFs
def process_pdfs(uploaded_files):
    """Extract text from PDFs and create a vector store"""
    text = ""
    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            if hasattr(uploaded_file, 'getbuffer'):
                temp_file.write(uploaded_file.getbuffer())
            else:
                temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        try:
            # Process the PDF file
            pdf_reader = PdfReader(temp_file_path)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            # Add file info to session state
            file_info = {
                "name": uploaded_file.name,
                "size": getattr(uploaded_file, 'size', len(uploaded_file.getvalue()))
            }
            st.session_state.uploaded_files_info.append(file_info)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create vector store with llama3:instruct embeddings
    try:
        embeddings = OllamaEmbeddings(
            base_url=st.session_state.ollama_api_base,
            model="llama3:instruct"  # Using the same model for embeddings
        )
        
        # Update or create vectorstore
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
        else:
            # Create a temporary vectorstore and merge with existing one
            temp_vectorstore = FAISS.from_texts(chunks, embeddings)
            st.session_state.vectorstore.merge_from(temp_vectorstore)
        
        # Initialize conversation chain with llama3:instruct
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key='answer'
        )
        
        st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOllama(
                model="llama3:instruct",
                base_url=st.session_state.ollama_api_base,
                temperature=0.7,
                system=(
                    "You are Dadhichi, an expert AI fitness and nutrition coach specializing in Indian lifestyle. "
                    "Provide specific, actionable advice for workouts, diet plans, and yoga routines. "
                    "Focus on practical solutions using common Indian foods and home exercises. "
                    "Be motivational but realistic. Break down complex concepts into simple steps. "
                    "When possible, reference and synthesize information from the uploaded documents."
                )
            ),
            retriever=st.session_state.vectorstore.as_retriever(),
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        
        return len(chunks)
    
    except Exception as e:
        st.error(f"Failed to create embeddings: {str(e)}")
        st.info("Please ensure 'llama3:instruct' model is available. Run: ollama pull llama3:instruct")
        return 0

# Function to load initial PDFs from directory
def load_initial_pdfs():
    """Load initial PDFs from a directory when the app starts"""
    pdf_directory = "initial_pdfs"  # Directory containing your PDFs
    
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory, exist_ok=True)
        return
    
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    if not pdf_files:
        return
    
    # Create proper file-like objects with all required attributes
    uploaded_files = []
    for pdf_file in pdf_files:
        file_path = os.path.join(pdf_directory, pdf_file)
        with open(file_path, 'rb') as f:
            file_content = f.read()
            
            file_obj = SimpleNamespace(
                name=pdf_file,
                size=len(file_content),
                getvalue=lambda: file_content,
                getbuffer=lambda: memoryview(file_content)
            )
            uploaded_files.append(file_obj)
    
    # Process the files
    if uploaded_files:
        with st.spinner("Loading initial PDFs..."):
            try:
                chunks_created = process_pdfs(uploaded_files)
                if chunks_created > 0:
                    st.sidebar.success(f"✅ Pre-loaded {len(uploaded_files)} PDFs with {chunks_created} chunks!")
            except Exception as e:
                st.sidebar.error(f"Error loading initial PDFs: {str(e)}")

# Function to generate response using llama3:instruct
def generate_response(input_text):
    if not input_text.strip():
        return "Please enter a valid question."
    
    try:
        if st.session_state.conversation_chain:
            response = st.session_state.conversation_chain({"question": input_text})
            return response['answer']
        else:
            # Fallback to basic llama3:instruct without vector store
            model = ChatOllama(
                model="llama3:instruct",
                base_url=st.session_state.ollama_api_base,
                temperature=0.7,
                system=(
                    "You are Dadhichi, an AI fitness coach for Indian users. "
                    "Provide specific fitness and nutrition advice using common "
                    "Indian foods and home exercises. Be practical and motivational."
                )
            )
            response = model.invoke(input_text)
            return response.content
            
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        st.error(error_msg)
        return error_msg

# Load initial PDFs when the app starts
if st.session_state.vectorstore is None:
    load_initial_pdfs()

# Sidebar configuration
with st.sidebar:
    st.header("Ollama Configuration")
    ollama_api = st.text_input("Ollama API URL", value=st.session_state.ollama_api_base)
    if ollama_api != st.session_state.ollama_api_base:
        st.session_state.ollama_api_base = ollama_api
        st.success("Ollama API URL updated!")
    
    st.markdown("""
    **Required Setup:**
    1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
    2. Pull the model: `ollama pull llama3:instruct`
    3. Start Ollama: `ollama serve`
    4. Ensure the server is running at the specified URL
    """)
    
    st.header("Knowledge Base")
    uploaded_files = st.file_uploader("Upload fitness PDFs", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and st.button("Process PDFs"):
        with st.spinner("Processing documents..."):
            try:
                chunks_created = process_pdfs(uploaded_files)
                if chunks_created > 0:
                    st.success(f"Processed {len(uploaded_files)} files with {chunks_created} chunks!")
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
    
    if st.session_state.uploaded_files_info:
        st.subheader("Loaded Documents")
        for file_info in st.session_state.uploaded_files_info:
            st.write(f"- {file_info['name']} ({file_info['size']/1024:.1f} KB)")
    
    if st.session_state.vectorstore and st.button("Clear Knowledge Base"):
        st.session_state.vectorstore = None
        st.session_state.conversation_chain = None
        st.session_state.uploaded_files_info = []
        st.success("Knowledge base cleared!")
        st.rerun()

# Main chat interface
st.subheader("Chat with Dadhichi")

# Display chat history
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for chat in reversed(st.session_state['chat_history']):
        if "Error" in chat["assistant"]:
            st.markdown(f'<div class="error-message"><strong>⚠️ Error:</strong> {chat["assistant"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="user-message"><strong>🧑 You:</strong> {chat["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="assistant-message"><strong>🧠 Dadhichi:</strong> {chat["assistant"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Chat input
with st.form("chat_form"):
    user_input = st.text_area("Ask about fitness, nutrition, or yoga:", height=100, key="user_input")
    submitted = st.form_submit_button("Send")
    
    if submitted and user_input:
        with st.spinner("Dadhichi is thinking..."):
            response = generate_response(user_input)
            st.session_state.chat_history.append({
                "user": user_input,
                "assistant": response
            })
            st.rerun()

# Footer
st.markdown("---")
st.write("""
    *Note: Dadhichi provides fitness guidance only, not medical advice. 
    For personalized health advice, consult a qualified professional.*
""")
