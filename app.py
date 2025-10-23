import streamlit as st
import requests
import uuid

API_URL= "http://0.0.0.0:8000"


def ask(query: str, collection_id: str) -> str:
    with st.spinner("Asking the chatbot..."):
        try:
            response = requests.post(f"{API_URL}/ask", json={"question": query, "collection_id": collection_id})
            if response.status_code == 200:
                data = response.json()
                return data["answer"]
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                st.error(error_msg)
                return "I couldn't find an answer to your question."
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API server. Please make sure the API is running.")
            return "I couldn't connect to the server."
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
            return "Request timed out. Please try again."
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return "An unexpected error occurred."

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

def initialize_session():
    """Initialize a new session with a unique collection ID"""
    if 'collection_id' not in st.session_state:
        st.session_state.collection_id = str(uuid.uuid4())
        st.session_state.session_initialized = True
        st.session_state.files_uploaded = False

def cleanup_session():
    """Clean up the current session"""
    if 'collection_id' in st.session_state:
        delete_collection_via_api(st.session_state.collection_id)
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    
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
    
    try:
        with st.spinner("Uploading files..."):
            response = requests.post(f"{API_URL}/feed/", files=files, data=data)
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
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API server. Please make sure the API is running.")
    except requests.exceptions.Timeout:
        st.error("Upload request timed out. Please try again.")
    except Exception as e:
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

# Chat interface
with st.chat_message(name="ai", avatar="ai"):
    st.write("Hello! I'm the Chatbot RAG. How can I help you today?")

query = st.chat_input(placeholder="Type your question here...")

if query:
    if not st.session_state.get('files_uploaded', False):
        st.error("Please upload PDF files first before asking questions.")
    else:
        with st.chat_message("user"):
            st.write(query)
        answer = ask(query, st.session_state.collection_id)
        with st.chat_message("ai"):
            st.write(answer)