import streamlit as st
import requests
import uuid
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Do NOT override environment provided by Docker; .env values only fill missing vars
load_dotenv(override=False)

HOST = os.environ.get("HOST", "localhost")
FASTAPI_PORT = os.environ.get("FASTAPI_PORT", "8000")

API_URL = f"http://{HOST}:{FASTAPI_PORT}"

print(f"[DEBUG] Connecting to API at: {API_URL}")


def test_api_connection(max_retries=3, delay=2):
    """Test if the API server is reachable"""
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{API_URL}/", timeout=5)
            if response.status_code == 200:
                print(f"[DEBUG] API connection successful on attempt {attempt + 1}")
                return True
        except Exception as e:
            print(f"[DEBUG] API connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    return False


def chat_with_history(query: str, collection_id: str, session_id: str) -> tuple[str, str]:
    """Enhanced chat function that maintains history"""
    with st.spinner("Asking the chatbot..."):
        try:
            # Get current chat history from session state
            chat_history = st.session_state.get('chat_history', [])

            # Prepare chat history for API - handle both datetime objects and strings
            history_for_api = []
            for msg in chat_history:
                timestamp = msg["timestamp"]
                # If it's already a string, use it as is; if it's a datetime object, convert it
                timestamp_str = timestamp if isinstance(timestamp, str) else timestamp.isoformat()
                
                history_for_api.append({
                    "role": msg["role"], 
                    "content": msg["content"], 
                    "timestamp": timestamp_str
                })

            response = requests.post(f"{API_URL}/chat", json={
                "question": query, 
                "collection_id": collection_id,
                "session_id": session_id,
                "chat_history": history_for_api
            })

            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                updated_history = data["chat_history"]
                
                # Update session state with new history
                st.session_state.chat_history = updated_history
                
                return answer, data["session_id"]
            else:
                error_msg = f"API Error , status code:{response.status_code}, error: {response.text}"
                st.error(error_msg)
                return "I couldn't find an answer to your question.", session_id
        except Exception as e:
            st.error(f"Unexpected error in chat with history: {str(e)}")
            return "An unexpected error occurred in chat with history.", session_id

def delete_collection_via_api(collection_id: str):
    """Delete a collection from the database via API"""
    try:
        response = requests.delete(f"{API_URL}/collection/{collection_id}")
        if response.status_code == 200:
            return True
        else:
            st.error(f"Failed to delete collection: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error deleting collection: {str(e)}")
        return False

def delete_session_and_collection_via_api(session_id: str):
    """Delete both session and associated collection via API"""
    try:
        response = requests.delete(f"{API_URL}/session/{session_id}")
        if response.status_code == 200:
            data = response.json()
            return data.get("session_cleared", False), data.get("collection_deleted", False)
        else:
            st.error(f"Failed to delete session: {response.text}")
            return False, False
    except Exception as e:
        st.error(f"Error deleting session: {str(e)}")
        return False, False

def initialize_session():
    """Initialize a new session with a unique collection ID and session ID"""
    if 'collection_id' not in st.session_state:
        st.session_state.collection_id = str(uuid.uuid4())
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.session_initialized = True
        st.session_state.files_uploaded = False


def cleanup_session():
    """Clean up the current session"""
    if 'session_id' in st.session_state:
        # Use the new API endpoint that deletes both session and collection
        session_cleared, collection_deleted = delete_session_and_collection_via_api(st.session_state.session_id)
        
        if session_cleared:
            st.success("✅ Session and collection cleaned up successfully!")
        else:
            st.warning("⚠️ Session cleanup may have failed")
        
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    elif 'collection_id' in st.session_state:
        # Fallback to old method if session_id is not available
        delete_collection_via_api(st.session_state.collection_id)
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
            
def clear_chat_history():
    """Clear the current chat history"""
    st.session_state.chat_history = []
    # Clear chat history on the server side
    if 'session_id' in st.session_state:
        try:
            response = requests.delete(f"{API_URL}/chat/{st.session_state.session_id}")
            if response.status_code == 200:
                st.success("Chat history cleared successfully!")
            else:
                st.warning("Chat history may not have been cleared on the server")
        except Exception as e:
            st.warning(f"Error clearing chat history on server: {str(e)}")

    
st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("Chatbot RAG")

# Initialize session
initialize_session()


uploaded_files = st.file_uploader("Upload your PDF documents", type="pdf", accept_multiple_files=True)

# Only process files if they haven't been uploaded yet
if uploaded_files and not st.session_state.get('files_uploaded', False):
    # Prepare files and data for multiple PDFs
    files = []
    data = []
    
    # Process each uploaded file
    for file in uploaded_files:
        # Create tuple for requests: (filename, file_content, content_type)
        files.append(('files', (file.name, file.getvalue(), 'application/pdf')))
        # Add filename as a separate form field
        data.append(('filepaths', file.name))
    
    # Add collection_id to the data
    data.append(('collection_id', st.session_state.collection_id))
    
    # First, test API connection
    if not test_api_connection():
        st.error("⚠️ Cannot connect to the API server. Please make sure the API is running and try again.")
        st.info(f"Attempting to connect to: {API_URL}")
        st.stop()
    
    try:
        with st.spinner("Uploading files..."):
            print(f"DEBUG: API_URL = {API_URL}")
            print(f"DEBUG: files = {files}")
            print(f"DEBUG: data = {data}")
            response = requests.post(f"{API_URL}/feed", files=files, data=data, timeout=300)
            print(f"DEBUG: Response status = {response.status_code}")
            print(f"DEBUG: Response text = {response.text[:200] if response.text else 'Empty response'}")
        if response.status_code == 200:
            data = response.json()
            # Display detailed upload statistics
            st.success("Files uploaded successfully!")
            
            # Mark files as uploaded in session state
            st.session_state.files_uploaded = True
            
            # Create columns for better layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Documents Processed", data.get("num_docs", 0))
            
            with col2:
                st.metric("Chunks Created", data.get("num_chunks", 0))
            
            with col3:
                status = data.get("status", "Unknown")
                if status == "Success":
                    st.success("Status: Success")
                else:
                    st.error(f"Status: {status}")
            
            # Display session info
            st.info(f"**Session ID:** {st.session_state.collection_id}")
            
            # Display processed files
            if data.get("files"):
                st.write("**Processed Files:**")
                for i, filename in enumerate(data["files"]):
                    filepath = data.get("files_path", [])[i] if i < len(data.get("files_path", [])) else filename
                    st.write(f"• {filename} (path: {filepath})")
            
        else:
            error_msg = f"Upload failed with status {response.status_code}: {response.text}"
            st.error(error_msg)
    except requests.exceptions.ConnectionError as e:
        print(f"DEBUG: ConnectionError - {e}")
        st.error("Cannot connect to the API server. Please make sure the API is running.")
    except requests.exceptions.Timeout as e:
        print(f"DEBUG: Timeout - {e}")
        st.error("Upload request timed out. Please try again.")
    except Exception as e:
        print(f"DEBUG: General Exception - {e}")
        st.error(f"Error uploading files: {str(e)}")

# Show upload status when files are already uploaded
elif st.session_state.get('files_uploaded', False):
    st.success("✅ Files already uploaded and ready for questions!")
    st.info(f"**Session ID:** {st.session_state.collection_id}")
        
# Session management controls
col1, col2 = st.columns([3, 1])
with col1:
    if st.session_state.get('files_uploaded', False):
        st.success("✅ Files uploaded and ready for questions!")
    else:
        st.warning("⚠️ Please upload PDF files first to start asking questions.")
with col2:
    if st.button("🔄 New Session", help="Start a new session (this will delete current collection)"):
        cleanup_session()
        st.rerun()

# Chat interface with history
if st.session_state.get('files_uploaded', False):
    # Display chat history
    if st.session_state.get('chat_history'):
        for msg in st.session_state.chat_history:
            with st.chat_message(name=msg["role"], avatar="ai" if msg["role"] == "assistant" else "user"):
                st.write(msg["content"])
    else:
        with st.chat_message(name="ai", avatar="ai"):
            st.write("Hello! I'm the Chatbot RAG. How can I help you today?")

    query = st.chat_input(placeholder="Type your question here...")

    if query:
        with st.chat_message("user"):
            st.write(query)
        
        answer, session_id = chat_with_history(query, st.session_state.collection_id, st.session_state.session_id)
        
        with st.chat_message("ai"):
            st.write(answer)
        
        # Add clear chat button
        if st.button("🗑️ Clear Chat History"):
            clear_chat_history()
            st.rerun()