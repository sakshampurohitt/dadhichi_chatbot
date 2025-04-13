import streamlit as st
import os
import tempfile
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
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🧠 Dadhichi</h1>', unsafe_allow_html=True)
st.write("I'm your personal AI fitness and nutrition coach powered by Llama 3. "
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
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name
        
        try:
            # Process the PDF file
            pdf_reader = PdfReader(temp_file_path)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            # Add file info to session state
            file_info = {
                "name": uploaded_file.name,
                "size": uploaded_file.size
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
    
    # Create vector store
    embeddings = OllamaEmbeddings(
        base_url=st.session_state.ollama_api_base,
        model="llama3"
    )
    
    # Update or create vectorstore
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
    else:
        # Create a temporary vectorstore and merge with existing one
        temp_vectorstore = FAISS.from_texts(chunks, embeddings)
        st.session_state.vectorstore.merge_from(temp_vectorstore)
    
    # Initialize conversation chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOllama(
            model="llama3:instruct",
            base_url=st.session_state.ollama_api_base,
            temperature=0.7,
            system=(
                "You are Dadhichi, a highly knowledgeable and supportive fitness and nutrition coach. "
                "You specialize in crafting personalized workout routines, balanced Indian diet plans, "
                "yoga practices for wellness, and providing motivational health tips. "
                "Always give clear, culturally relevant suggestions focused on exercise, fitness, "
                "yoga, hydration, sleep, and nutrition. Avoid medical advice or diagnosing conditions. "
                "Act like a real coach who understands the daily life and diet habits of people in India. "
                "Use the knowledge from the uploaded PDF documents to provide more accurate and specialized advice."
            )
        ),
        retriever=st.session_state.vectorstore.as_retriever(),
        memory=memory
    )
    
    return len(chunks)

# Function to generate response using conversation chain if available, or regular ChatOllama
def generate_response(input_text):
    if st.session_state.conversation_chain:
        # Use the conversation chain with vector store
        response = st.session_state.conversation_chain({"question": input_text})
        return response['answer']
    else:
        # Use regular ChatOllama
        model = ChatOllama(
            model="llama3:instruct",
            base_url=st.session_state.ollama_api_base,
            temperature=0.7,
            system=(
                "You are Dadhichi, a highly knowledgeable and supportive fitness and nutrition coach. "
                "You specialize in crafting personalized workout routines, balanced Indian diet plans, "
                "yoga practices for wellness, and providing motivational health tips. "
                "Always give clear, culturally relevant suggestions focused on exercise, fitness, "
                "yoga, hydration, sleep, and nutrition. Avoid medical advice or diagnosing conditions. "
                "Act like a real coach who understands the daily life and diet habits of people in India."
            )
        )
        response = model.invoke(input_text)
        return response.content

# Sidebar for PDF uploads and status
with st.sidebar:
    st.header("Ollama Configuration")
    ollama_api = st.text_input("Ollama API URL", value=st.session_state.ollama_api_base)
    if ollama_api != st.session_state.ollama_api_base:
        st.session_state.ollama_api_base = ollama_api
        st.success("Ollama API URL updated!")
    
    st.markdown("""
    **How to connect to your local Ollama:**
    1. Install Ollama from [ollama.ai](https://ollama.ai)
    2. Run `ollama pull llama3` and `ollama pull llama3:instruct`
    3. Start Ollama server locally
    4. Keep URL as `http://localhost:11434` for local use
    5. For remote use, you'll need to expose your Ollama API securely
    """)
    
    with st.expander("Troubleshooting Connection"):
        st.markdown("""
        - Make sure Ollama is running on your machine
        - If running remotely, make sure the API is accessible from this Replit app
        - Check if model `llama3` and `llama3:instruct` are downloaded
        - Try restarting Ollama if connection issues persist
        """)
        
    st.header("Knowledge Base")
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process PDFs"):
            with st.spinner("Processing PDF documents..."):
                try:
                    chunks_created = process_pdfs(uploaded_files)
                    st.success(f"✅ Successfully processed {len(uploaded_files)} PDFs and created {chunks_created} text chunks!")
                except Exception as e:
                    error_message = str(e)
                    if "Cannot assign requested address" in error_message or "Connection" in error_message:
                        st.error("⚠️ Failed to connect to Ollama API. Please check your Ollama server is running and accessible.")
                        st.info("See the 'Troubleshooting Connection' section above for help.")
                    else:
                        st.error(f"Error processing PDFs: {error_message}")
    
    # Display uploaded files info
    if st.session_state.uploaded_files_info:
        st.subheader("Uploaded Documents")
        for idx, file_info in enumerate(st.session_state.uploaded_files_info):
            file_size_kb = round(file_info["size"] / 1024, 2)
            st.write(f"{idx+1}. {file_info['name']} ({file_size_kb} KB)")
    
    # Clear knowledge base button
    if st.session_state.vectorstore and st.button("Clear Knowledge Base"):
        st.session_state.vectorstore = None
        st.session_state.conversation_chain = None
        st.session_state.uploaded_files_info = []
        st.success("Knowledge base cleared!")
        st.rerun()

# Main chat interface
st.subheader("Chat with Dadhichi")
# Display chat input form
with st.form("llm-form", clear_on_submit=True):
    text = st.text_area("Enter your question or statement:", height=100)
    submit = st.form_submit_button("Submit")

# On form submission
if submit and text:
    with st.spinner("Generating response..."):
        try:
            response = generate_response(text)
            st.session_state['chat_history'].append({"user": text, "assistant": response})
        except Exception as e:
            error_message = str(e)
            if "Cannot assign requested address" in error_message or "Connection" in error_message:
                st.error("⚠️ Failed to connect to Ollama API. Please check your Ollama server is running and accessible.")
                st.info("See the 'Troubleshooting Connection' section in the sidebar for help.")
            else:
                st.error(f"Error generating response: {error_message}")
            # Add error to history so user knows what happened
            st.session_state['chat_history'].append({
                "user": text, 
                "assistant": "⚠️ Error: Could not connect to Ollama. Please check the Ollama server is running and the API URL is correct."
            })

# Display chat history
chat_container = st.container()
with chat_container:
    for chat in reversed(st.session_state['chat_history']):
        st.markdown(f"**🧑 User**: {chat['user']}")
        st.markdown(f"**🧠 Dadhichi**: {chat['assistant']}")
        st.markdown("---")

# Footer
st.write("📝 *Note: Dadhichi is powered by Llama 3 and LangChain. For fitness guidance only, not medical advice.*")
