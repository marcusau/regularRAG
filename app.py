"""
Streamlit web application for RAG (Retrieval-Augmented Generation) chatbot.

This module provides a user-friendly web interface for interacting with the RAG chatbot.
It handles document uploads, chat interactions, and session management through a
Streamlit-based frontend that communicates with the FastAPI backend.

Key Features:
- PDF document upload and processing
- Interactive chat interface with conversation history
- Session management and cleanup
- Real-time API connectivity testing
- User-friendly error handling and status messages

Author: RAG Development Team
Version: 0.1
"""

import os
import time
import uuid
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables - do NOT override environment provided by Docker
# .env values only fill missing variables
load_dotenv(override=False)

# Configuration from environment variables
HOST = os.environ.get("HOST", "localhost")  # API server host
FASTAPI_PORT = os.environ.get("FASTAPI_PORT", "8000")  # API server port

# Construct API base URL
API_URL = f"http://{HOST}:{FASTAPI_PORT}"



def test_api_connection(max_retries=3, delay=2):
    """
    Test if the API server is reachable and responding.
    
    This function attempts to connect to the FastAPI backend server with
    retry logic to handle temporary connection issues. It's used to verify
    that the API is running before attempting to upload documents or send
    chat requests.
    
    Args:
        max_retries (int, optional): Maximum number of connection attempts. Defaults to 3.
        delay (int, optional): Delay in seconds between retry attempts. Defaults to 2.
    
    Returns:
        bool: True if connection is successful, False otherwise
    
    Process:
        1. Make GET request to API root endpoint
        2. Check for 200 status code response
        3. Retry with exponential backoff if connection fails
        4. Log all attempts for debugging
    
    Example:
        >>> if test_api_connection():
        ...     print("API is ready")
        ... else:
        ...     print("API is not available")
    """
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
    """
    Enhanced chat function that maintains conversation history.
    
    This function handles chat interactions by sending user queries to the
    FastAPI backend while maintaining conversation context. It processes
    the response and updates the Streamlit session state with the new
    conversation history.
    
    Args:
        query (str): The user's question or message
        collection_id (str): Vector collection ID for document search
        session_id (str): Session ID for conversation continuity
    
    Returns:
        tuple[str, str]: A tuple containing:
            - answer: The chatbot's response
            - session_id: The session ID (may be updated by the API)
    
    Process:
        1. Retrieve existing chat history from Streamlit session state
        2. Convert timestamps to ISO format for API compatibility
        3. Send POST request to /chat endpoint with query and history
        4. Process response and update session state
        5. Return answer and session ID
    
    Error Handling:
        - Displays user-friendly error messages in Streamlit
        - Returns fallback responses for API errors
        - Handles both network and processing errors gracefully
    
    Note:
        This function updates st.session_state.chat_history with the
        complete conversation history returned by the API.
    
    Example:
        >>> answer, session_id = chat_with_history(
        ...     "What is the main topic?",
        ...     "collection-123",
        ...     "session-456"
        ... )
        >>> print(answer)
    """
    with st.spinner("Asking the chatbot..."):
        try:
            # Get current chat history from Streamlit session state
            chat_history = st.session_state.get("chat_history", [])

            # Prepare chat history for API - handle both datetime objects and strings
            history_for_api = []
            for msg in chat_history:
                timestamp = msg["timestamp"]
                # Convert datetime objects to ISO format strings for API compatibility
                timestamp_str = timestamp if isinstance(timestamp, str) else timestamp.isoformat()

                history_for_api.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                        "timestamp": timestamp_str,
                    }
                )

            # Send chat request to FastAPI backend
            response = requests.post(
                f"{API_URL}/chat",
                json={
                    "question": query,
                    "collection_id": collection_id,
                    "session_id": session_id,
                    "chat_history": history_for_api,
                },
            )

            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                updated_history = data["chat_history"]

                # Update Streamlit session state with new conversation history
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
    """
    Delete a vector collection from the database via API.
    
    This function sends a DELETE request to the FastAPI backend to remove
    a specific vector collection. It's used for cleanup operations when
    sessions are ended or collections are no longer needed.
    
    Args:
        collection_id (str): The unique identifier of the collection to delete
    
    Returns:
        bool: True if deletion was successful, False otherwise
    
    Process:
        1. Send DELETE request to /collection/{collection_id} endpoint
        2. Check response status code
        3. Display appropriate success/error messages
        4. Return boolean result for further processing
    
    Error Handling:
        - Displays error messages in Streamlit interface
        - Returns False for any failure (network or API errors)
        - Logs errors for debugging purposes
    
    Example:
        >>> if delete_collection_via_api("collection-123"):
        ...     print("Collection deleted successfully")
        ... else:
        ...     print("Failed to delete collection")
    """
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
    """
    Delete both session and associated collection via API.
    
    This function performs a complete cleanup by deleting both the chat session
    and its associated vector collection from the backend. It's used when
    starting a new session or performing manual cleanup operations.
    
    Args:
        session_id (str): The unique identifier of the session to delete
    
    Returns:
        tuple[bool, bool]: A tuple containing:
            - session_cleared: True if session was successfully deleted
            - collection_deleted: True if associated collection was deleted
    
    Process:
        1. Send DELETE request to /session/{session_id} endpoint
        2. Parse response to extract deletion status
        3. Display appropriate success/error messages
        4. Return boolean results for both operations
    
    Error Handling:
        - Displays error messages in Streamlit interface
        - Returns (False, False) for any failure
        - Handles both network and API errors gracefully
    
    Note:
        The API endpoint handles both session and collection deletion
        in a single operation, ensuring data consistency.
    
    Example:
        >>> session_cleared, collection_deleted = delete_session_and_collection_via_api("session-123")
        >>> if session_cleared and collection_deleted:
        ...     print("Complete cleanup successful")
    """
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
    """
    Initialize a new session with unique identifiers and default state.
    
    This function sets up a new Streamlit session by generating unique IDs
    for both the collection and session, and initializing the session state
    with default values. It ensures that each user session has its own
    isolated data and conversation history.
    
    Session State Variables:
        - collection_id: Unique identifier for the vector collection
        - session_id: Unique identifier for the chat session
        - chat_history: List of conversation messages
        - session_initialized: Flag indicating session setup completion
        - files_uploaded: Flag indicating if documents have been uploaded
    
    Process:
        1. Check if session is already initialized
        2. Generate unique UUIDs for collection and session
        3. Initialize empty chat history
        4. Set initialization flags
        5. Mark files as not uploaded initially
    
    Note:
        This function is idempotent - it only initializes if not already done.
        It's called at the start of the Streamlit app to ensure proper setup.
    
    Example:
        >>> initialize_session()
        >>> print(st.session_state.collection_id)  # UUID string
        >>> print(st.session_state.session_id)     # UUID string
    """
    if "collection_id" not in st.session_state:
        # Generate unique identifiers for this session
        st.session_state.collection_id = str(uuid.uuid4())
        st.session_state.session_id = str(uuid.uuid4())
        
        # Initialize session state with default values
        st.session_state.chat_history = []
        st.session_state.session_initialized = True
        st.session_state.files_uploaded = False


def cleanup_session():
    """
    Clean up the current session and associated data.
    
    This function performs a complete cleanup of the current Streamlit session
    by deleting both the backend session and its associated vector collection,
    then clearing all local session state. It's used when starting a new
    session or when the user explicitly requests cleanup.
    
    Process:
        1. Check if session_id exists in session state
        2. Call API to delete both session and collection
        3. Display success/warning messages based on results
        4. Clear all session state variables
        5. Fallback to collection-only deletion if session_id missing
    
    Cleanup Operations:
        - Deletes backend session and collection via API
        - Removes all Streamlit session state variables
        - Provides user feedback on cleanup success/failure
        - Handles both new and legacy session formats
    
    User Feedback:
        - Success message if both session and collection deleted
        - Warning message if cleanup may have failed
        - Automatic fallback for older session formats
    
    Note:
        This function completely resets the session state, requiring
        re-initialization for further operations.
    
    Example:
        >>> cleanup_session()
        >>> # Session state is now completely cleared
        >>> # User will need to upload files again
    """
    if "session_id" in st.session_state:
        # Use the comprehensive API endpoint that deletes both session and collection
        session_cleared, collection_deleted = delete_session_and_collection_via_api(st.session_state.session_id)

        if session_cleared:
            st.success("✅ Session and collection cleaned up successfully!")
        else:
            st.warning("⚠️ Session cleanup may have failed")

        # Clear all session state variables
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    elif "collection_id" in st.session_state:
        # Fallback to collection-only deletion for older session formats
        delete_collection_via_api(st.session_state.collection_id)
        # Clear all session state variables
        for key in list(st.session_state.keys()):
            del st.session_state[key]


def clear_chat_history():
    """
    Clear the current chat history both locally and on the server.
    
    This function removes all conversation history from both the local
    Streamlit session state and the backend server. It's used when
    users want to start a fresh conversation within the same session
    without losing their uploaded documents.
    
    Process:
        1. Clear local chat history from session state
        2. Send DELETE request to clear server-side history
        3. Display appropriate success/warning messages
        4. Handle any errors gracefully
    
    Server-Side Cleanup:
        - Sends DELETE request to /chat/{session_id} endpoint
        - Only attempts if session_id is available
        - Provides user feedback on server operation success
    
    Local Cleanup:
        - Immediately clears st.session_state.chat_history
        - Ensures UI reflects cleared state immediately
    
    Error Handling:
        - Displays warning messages for server-side failures
        - Continues operation even if server cleanup fails
        - Logs errors for debugging purposes
    
    Note:
        This function preserves the session and collection data,
        only clearing the conversation history.
    
    Example:
        >>> clear_chat_history()
        >>> # Chat history is now empty both locally and on server
        >>> # User can start a new conversation with same documents
    """
    # Clear local chat history immediately
    st.session_state.chat_history = []
    
    # Clear chat history on the server side
    if "session_id" in st.session_state:
        try:
            response = requests.delete(f"{API_URL}/chat/{st.session_state.session_id}")
            if response.status_code == 200:
                st.success("Chat history cleared successfully!")
            else:
                st.warning("Chat history may not have been cleared on the server")
        except Exception as e:
            st.warning(f"Error clearing chat history on server: {str(e)}")


# Configure Streamlit page
st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("Chatbot RAG")

# Initialize session with unique identifiers and default state
initialize_session()


# File upload interface - allows users to upload multiple PDF documents
uploaded_files = st.file_uploader("Upload your PDF documents", type="pdf", accept_multiple_files=True)

# Only process files if they haven't been uploaded yet to avoid duplicate processing
if uploaded_files and not st.session_state.get("files_uploaded", False):
    # Prepare files and data for multiple PDF upload to FastAPI backend
    files = []
    data = []

    # Process each uploaded file and prepare for multipart form data
    for file in uploaded_files:
        # Create tuple for requests: (field_name, (filename, file_content, content_type))
        files.append(("files", (file.name, file.getvalue(), "application/pdf")))
        # Add filename as a separate form field for the API
        data.append(("filepaths", file.name))

    # Add collection_id to the form data for document storage
    data.append(("collection_id", st.session_state.collection_id))

    # Test API connection before attempting upload
    if not test_api_connection():
        st.error("⚠️ Cannot connect to the API server. Please make sure the API is running and try again.")
        st.info(f"Attempting to connect to: {API_URL}")
        st.stop()

    try:
        with st.spinner("Uploading files..."):
            response = requests.post(f"{API_URL}/feed", files=files, data=data, timeout=300)
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
elif st.session_state.get("files_uploaded", False):
    st.success("✅ Files already uploaded and ready for questions!")
    st.info(f"**Session ID:** {st.session_state.collection_id}")

# Session management controls - display status and provide new session option
col1, col2 = st.columns([3, 1])
with col1:
    # Display current session status
    if st.session_state.get("files_uploaded", False):
        st.success("✅ Files uploaded and ready for questions!")
    else:
        st.warning("⚠️ Please upload PDF files first to start asking questions.")
with col2:
    # New session button - cleans up current session and starts fresh
    if st.button(
        "🔄 New Session",
        help="Start a new session (this will delete current collection)",
    ):
        cleanup_session()
        st.rerun()

# Chat interface with conversation history - only available after files are uploaded
if st.session_state.get("files_uploaded", False):
    # Display existing chat history in conversation format
    if st.session_state.get("chat_history"):
        for msg in st.session_state.chat_history:
            with st.chat_message(name=msg["role"], avatar="ai" if msg["role"] == "assistant" else "user"):
                st.write(msg["content"])
    else:
        # Display welcome message for new conversations
        with st.chat_message(name="ai", avatar="ai"):
            st.write("Hello! I'm the Chatbot RAG. How can I help you today?")

    # Chat input widget for user questions
    query = st.chat_input(placeholder="Type your question here...")

    if query:
        # Display user's question in chat interface
        with st.chat_message("user"):
            st.write(query)

        # Process question through RAG pipeline and get response
        answer, session_id = chat_with_history(query, st.session_state.collection_id, st.session_state.session_id)

        # Display chatbot's response in chat interface
        with st.chat_message("ai"):
            st.write(answer)

        # Clear chat history button - allows users to start fresh conversation
        if st.button("🗑️ Clear Chat History"):
            clear_chat_history()
            st.rerun()
