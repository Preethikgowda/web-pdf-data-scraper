import streamlit as st
import os
from dotenv import load_dotenv
import time
import tempfile
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SmartDocAI:
    """Professional Document AI Assistant with enhanced features and error handling."""
    
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.setup_page_config()
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="SmartDoc AI - Professional Document Assistant",
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def initialize_session_state(self):
        """Initialize session state variables."""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = None
        if "processed_documents" not in st.session_state:
            st.session_state.processed_documents = []
        if "processing_status" not in st.session_state:
            st.session_state.processing_status = "idle"
            
    def apply_custom_css(self):
        """Apply custom CSS styling using Tailwind-inspired classes."""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        
        .sidebar-section {
            background: #f8fafc;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border: 1px solid #e2e8f0;
        }
        
        .chat-message {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .user-message {
            background: #f0f9ff;
            border-left-color: #0ea5e9;
        }
        
        .ai-message {
            background: #f0fdf4;
            border-left-color: #10b981;
        }
        
        .status-success {
            background: #dcfce7;
            color: #166534;
            padding: 0.75rem;
            border-radius: 0.375rem;
            border: 1px solid #bbf7d0;
        }
        
        .status-error {
            background: #fef2f2;
            color: #991b1b;
            padding: 0.75rem;
            border-radius: 0.375rem;
            border: 1px solid #fecaca;
        }
        
        .status-warning {
            background: #fffbeb;
            color: #92400e;
            padding: 0.75rem;
            border-radius: 0.375rem;
            border: 1px solid #fed7aa;
        }
        
        .document-chip {
            background: #e0e7ff;
            color: #3730a3;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            margin: 0.25rem;
            display: inline-block;
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border: 1px solid #e2e8f0;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .processing-spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """, unsafe_allow_html=True)
        
    def render_header(self):
        """Render the main application header."""
        st.markdown("""
        <div class="main-header">
            <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">📄 SmartDoc AI</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                Professional Document Intelligence Assistant
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    def render_sidebar(self):
        """Render the enhanced sidebar with professional styling."""
        st.sidebar.markdown("""
        <div class="sidebar-section">
            <h3 style="margin: 0 0 1rem 0; color: #374151;">🔍 Document Sources</h3>
            <p style="margin: 0; color: #6b7280; font-size: 0.9rem;">
                Choose your preferred method to input documents for analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Data source selection
        option = st.sidebar.radio(
            "Select Data Source:",
            ["Web URL", "Upload PDF"],
            index=0,
            help="Choose between web scraping or PDF upload"
        )
        
        uploaded_docs = None
        url_input = None
        
        if option == "Web URL":
            st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            url_input = st.sidebar.text_input(
                "🌐 Enter URL",
                placeholder="https://example.com",
                help="Enter a valid URL to scrape content from"
            )
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
        elif option == "Upload PDF":
            st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            uploaded_docs = st.sidebar.file_uploader(
                "📄 Upload PDF Files",
                type=["pdf"],
                accept_multiple_files=True,
                help="Upload one or more PDF files for analysis"
            )
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
        return option, url_input, uploaded_docs
        
    def render_chat_management(self):
        """Render chat history management section."""
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        <div class="sidebar-section">
            <h3 style="margin: 0 0 1rem 0; color: #374151;">💬 Chat Management</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🧹 Clear All", use_container_width=True):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
                
        with col2:
            chat_count = len(st.session_state.chat_history)
            st.metric("Messages", chat_count)
            
        if st.session_state.chat_history:
            selected_question = st.sidebar.selectbox(
                "🗂 Select chat to delete",
                [f"{chat['question'][:50]}..." if len(chat['question']) > 50 
                 else chat['question'] for chat in st.session_state.chat_history],
                help="Select a specific chat message to delete"
            )
            
            if st.sidebar.button("❌ Delete Selected", use_container_width=True):
                original_question = next(
                    (chat['question'] for chat in st.session_state.chat_history 
                     if chat['question'].startswith(selected_question.replace("...", ""))),
                    None
                )
                if original_question:
                    st.session_state.chat_history = [
                        chat for chat in st.session_state.chat_history 
                        if chat['question'] != original_question
                    ]
                    st.success("Selected chat deleted!")
                    
    def validate_inputs(self, option: str, url_input: Optional[str], uploaded_docs: Optional[List]) -> bool:
        """Validate user inputs before processing."""
        if option == "Web URL":
            if not url_input:
                return False
            if not url_input.startswith(('http://', 'https://')):
                st.error("Please enter a valid URL starting with http:// or https://")
                return False
        elif option == "Upload PDF":
            if not uploaded_docs:
                return False
        return True
        
    def process_documents(self, option: str, url_input: Optional[str], uploaded_docs: Optional[List]) -> bool:
        """Process documents with enhanced error handling and user feedback."""
        try:
            st.session_state.processing_status = "processing"
            
            with st.spinner("Processing documents..."):
                if option == "Web URL":
                    loader = WebBaseLoader(url_input)
                    docs = loader.load()
                    st.session_state.processed_documents = [url_input]
                    
                elif option == "Upload PDF":
                    docs = []
                    document_names = []
                    
                    for pdf in uploaded_docs:
                        document_names.append(pdf.name)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(pdf.read())
                            tmp_file_path = tmp_file.name
                            
                        loader = PyPDFLoader(tmp_file_path)
                        docs.extend(loader.load())
                        
                        # Clean up temporary file
                        os.unlink(tmp_file_path)
                        
                    st.session_state.processed_documents = document_names
                
                # Process documents
                if docs:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200
                    )
                    chunks = splitter.split_documents(docs[:50])
                    st.session_state.vectorstore = FAISS.from_documents(chunks, self.embedding_model)
                    st.session_state.processing_status = "success"
                    return True
                else:
                    st.session_state.processing_status = "error"
                    return False
                    
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            st.session_state.processing_status = "error"
            st.error(f"Error processing documents: {str(e)}")
            return False
            
    def render_document_status(self):
        """Render document processing status."""
        if st.session_state.processing_status == "success":
            st.markdown("""
            <div class="status-success">
                ✅ <strong>Documents processed successfully!</strong><br>
                Ready to answer your questions.
            </div>
            """, unsafe_allow_html=True)
            
            # Show processed documents
            if st.session_state.processed_documents:
                st.markdown("**Processed Documents:**")
                for doc in st.session_state.processed_documents:
                    st.markdown(f'<span class="document-chip">{doc}</span>', unsafe_allow_html=True)
                    
        elif st.session_state.processing_status == "error":
            st.markdown("""
            <div class="status-error">
                ❌ <strong>Error processing documents!</strong><br>
                Please check your inputs and try again.
            </div>
            """, unsafe_allow_html=True)
            
    def handle_query(self, prompt: str) -> Dict[str, Any]:
        """Handle user queries with enhanced response formatting."""
        try:
            if not self.groq_api_key:
                raise ValueError("GROQ API key not found. Please set GROQ_API_KEY environment variable.")
                
            llm = ChatGroq(groq_api_key=self.groq_api_key, model_name="llama3-8b-8192")
            
            prompt_template = ChatPromptTemplate.from_template("""
            You are a professional document analysis assistant. Answer questions based on the provided context only.
            Provide accurate, well-structured responses with proper formatting when appropriate.
            
            <context>
            {context}
            </context>
            
            Question: {input}
            
            Please provide a comprehensive answer based on the context above.
            """)
            
            document_chain = create_stuff_documents_chain(llm, prompt_template)
            retriever = st.session_state.vectorstore.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            start_time = time.process_time()
            response = retrieval_chain.invoke({"input": prompt})
            response_time = round(time.process_time() - start_time, 2)
            
            return {
                "answer": response["answer"],
                "context": response["context"],
                "response_time": response_time,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sources": st.session_state.processed_documents
            }
            
        except Exception as e:
            logger.error(f"Error handling query: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
    def render_chat_interface(self):
        """Render the enhanced chat interface."""
        st.markdown("---")
        st.subheader("💬 Ask Your Question")
        
        prompt = st.text_input(
            "🤔 What would you like to know?",
            placeholder="Enter your question about the documents...",
            key="user_input"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            ask_button = st.button("🚀 Ask Question", use_container_width=True)
        with col2:
            st.metric("Response Time", f"{getattr(st.session_state, 'last_response_time', 0)}s")
            
        if ask_button and prompt:
            response = self.handle_query(prompt)
            
            if "error" in response:
                st.error(f"Error: {response['error']}")
            else:
                # Display response
                st.markdown(f"""
                <div class="ai-message">
                    <strong>🤖 AI Assistant:</strong><br>
                    {response['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.last_response_time = response['response_time']
                
                # Save to chat history
                st.session_state.chat_history.append({
                    "question": prompt,
                    "answer": response['answer'],
                    "sources": response['sources'],
                    "timestamp": response['timestamp'],
                    "response_time": response['response_time']
                })
                
                # Show context in expander
                with st.expander("📄 View Source Context"):
                    for i, doc in enumerate(response['context']):
                        st.markdown(f"**Context {i+1}:**")
                        st.text(doc.page_content)
                        st.markdown("---")
                        
    def render_chat_history(self):
        """Render enhanced chat history."""
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("📊 Chat History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                st.markdown(f"""
                <div class="chat-message">
                    <div class="user-message">
                        <strong>👤 You:</strong> {chat['question']}
                    </div>
                    <div class="ai-message" style="margin-top: 1rem;">
                        <strong>🤖 AI:</strong> {chat['answer']}
                    </div>
                    <div style="margin-top: 1rem; font-size: 0.8rem; color: #6b7280;">
                        🕒 {chat.get('timestamp', 'Unknown')} | 
                        ⚡ {chat.get('response_time', 0)}s | 
                        📄 Sources: {', '.join(chat.get('sources', []))}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
    def render_empty_state(self):
        """Render empty state when no documents are loaded."""
        st.markdown("""
        <div class="status-warning">
            <h3 style="margin: 0 0 1rem 0;">🚀 Get Started</h3>
            <p style="margin: 0;">
                Please provide a valid URL or upload at least one PDF file to begin using SmartDoc AI.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show features
        st.markdown("### ✨ Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>🌐 Web Scraping</h4>
                <p>Extract content from any public webpage</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>📄 PDF Analysis</h4>
                <p>Upload and analyze multiple PDF documents</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>💬 Smart Chat</h4>
                <p>Ask questions and get intelligent responses</p>
            </div>
            """, unsafe_allow_html=True)
            
    def run(self):
        """Main application runner."""
        self.apply_custom_css()
        self.render_header()
        
        # Sidebar
        option, url_input, uploaded_docs = self.render_sidebar()
        self.render_chat_management()
        
        # Main content
        if self.validate_inputs(option, url_input, uploaded_docs):
            if st.session_state.processing_status == "idle":
                if self.process_documents(option, url_input, uploaded_docs):
                    st.rerun()
                    
            self.render_document_status()
            
            if st.session_state.vectorstore is not None:
                self.render_chat_interface()
                self.render_chat_history()
        else:
            self.render_empty_state()

# Application entry point
if __name__ == "__main__":
    app = SmartDocAI()
    app.run()