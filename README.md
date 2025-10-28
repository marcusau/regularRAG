# RAG Chatbot with FastAPI and Streamlit

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions about their content. The system uses ChromaDB for vector storage, Ollama for embeddings, and XAI for language generation.

## Features

- 📄 **PDF Document Processing**: Upload and process multiple PDF files
- 🔍 **Semantic Search**: Find relevant content using vector similarity search
- 🤖 **AI-Powered Q&A**: Get intelligent answers based on your documents
- 🌐 **Web Interface**: User-friendly Streamlit interface
- 🚀 **FastAPI Backend**: RESTful API for document processing and querying
- 💾 **Persistent Storage**: ChromaDB for vector storage and retrieval

## Architecture

- **Frontend**: Streamlit web interface
- **Backend**: FastAPI REST API
- **Vector Database**: ChromaDB for document embeddings
- **Embeddings**: Ollama (local embedding models)
- **LLM**: XAI for text generation

## Prerequisites

- Python 3.8+
- Ollama installed and running
- XAI API key

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd secondRAG
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Setup Ollama

1. **Install Ollama**:
   - Visit [https://ollama.ai](https://ollama.ai) and follow installation instructions for your OS

2. **Pull Embedding Model**:
   ```bash
   ollama pull nomic-embed-text  # or another embedding model of your choice
   ```

### 5. Environment Configuration

Create a `.env` file in the project root with the following variables:

```env
# XAI Configuration
XAI_API_KEY=your_xai_api_key_here
XAI_MODEL=your_xai_model_name

# Ollama Configuration
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# ChromaDB Configuration
CHROMADB_LOCAL_DIR=./chromadb
```

**Important**: Replace the placeholder values with your actual:
- XAI API key (get from [https://x.ai](https://x.ai))
- XAI model name (e.g., "grok-beta")
- Ollama embedding model name (e.g., "nomic-embed-text")

## Usage

### Option 1: Using Docker (Recommended)

The easiest way to run the application is using Docker Compose.

#### Quick Start

1. **Verify your setup**:
   ```bash
   ./check_setup.sh
   ```
   This will check if all required environment variables and dependencies are configured.

2. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

3. **Access the application**:
   - **Streamlit UI**: http://localhost:8501
   - **FastAPI Backend**: http://localhost:8000
   - **API Docs**: http://localhost:8000/docs

For detailed Docker setup instructions, see [DOCKER_SETUP.md](DOCKER_SETUP.md).

### Option 2: Running Locally

#### 1. Start the FastAPI Backend

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### 2. Start the Streamlit Frontend

```bash
STREAMLIT_BROWSER_GATHERUSAGESTATS=false STREAMLIT_SERVER_HEADLESS=true streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

The web interface will be available at `http://localhost:8501`

#### 3. Using the Application

1. **Upload Documents**: Use the file uploader to upload one or more PDF files
2. **Wait for Processing**: The system will process and vectorize your documents
3. **Ask Questions**: Type your questions in the chat interface
4. **Get Answers**: Receive AI-generated answers based on your document content

## API Endpoints

- `GET /` - Health check
- `POST /feed` - Upload and process PDF documents
- `POST /ask` - Ask questions about uploaded documents
- `POST /documents` - Search for relevant documents
- `DELETE /collection/{collection_id}` - Delete a document collection

## Project Structure

```
secondRAG/
├── api.py                 # FastAPI backend
├── app.py                 # Streamlit frontend
├── models.py              # Pydantic models
├── fastapi_models.py      # FastAPI response models
├── providers.py           # LLM and embedding providers
├── vector_store.py        # ChromaDB operations
├── rag_processor.py       # RAG query processing
├── preprocess.py          # Document parsing and chunking
├── utils.py               # Utility functions
├── prompts/               # Prompt templates
│   ├── generator/
│   └── rerank/
├── data/                  # Sample PDF documents
├── chromadb/              # ChromaDB storage directory
└── requirements.txt       # Python dependencies
```

## Configuration

### Supported File Types
- PDF documents only

### Embedding Models
The system uses Ollama for embeddings. Popular models include:
- `nomic-embed-text` (recommended)
- `all-minilm`
- `mxbai-embed-large`

### LLM Models
Configure your XAI model in the `.env` file. Examples:
- `grok-beta`
- `grok-2`

## Troubleshooting

### Docker-Specific Issues

1. **"Cannot connect to the API server" in Streamlit UI**:
   - **Cause**: Missing environment variables or FastAPI container not starting
   - **Solution**: 
     - Run `./check_setup.sh` to verify setup
     - Ensure you have a `.env` file with `XAI_API_KEY` and other required variables
     - Check FastAPI logs: `docker-compose logs fastapi`
     - Verify Ollama is running: `ollama serve`

2. **FastAPI container keeps restarting**:
   - **Cause**: Missing dependencies or configuration
   - **Solution**: 
     - Check logs: `docker-compose logs fastapi`
     - Verify `XAI_API_KEY` is set in `.env`
     - Ensure Ollama is accessible at the configured URL

3. **Port already in use**:
   - **Solution**: Stop services using ports 8000 or 8501, or modify port mappings in `docker-compose.yml`

4. **Ollama models not found**:
   - **Solution**: Pull the required model: `ollama pull nomic-embed-text`

For detailed Docker troubleshooting, see [DOCKER_SETUP.md](DOCKER_SETUP.md).

### Common Issues

1. **Ollama Connection Error**:
   - Ensure Ollama is running: `ollama serve`
   - Check if the embedding model is pulled: `ollama list`

2. **XAI API Error**:
   - Verify your API key is correct
   - Check your XAI account credits

3. **ChromaDB Issues**:
   - Ensure the `CHROMADB_LOCAL_DIR` path exists and is writable
   - Check disk space for vector storage

4. **Port Conflicts**:
   - Change ports in the startup commands if 8000 or 8501 are in use

### Logs and Debugging

- FastAPI logs are displayed in the terminal where you started the API server
- Streamlit logs are shown in the terminal where you started the Streamlit app
- Check the browser console for frontend errors

## Development

### Adding New Features

1. **New Document Types**: Modify `preprocess.py` to support additional file formats
2. **Custom Embeddings**: Update `providers.py` to use different embedding models
3. **UI Improvements**: Modify `app.py` for frontend changes
4. **API Extensions**: Add new endpoints in `api.py`

### Testing

Test the API endpoints using curl or a tool like Postman:

```bash
# Health check
curl http://localhost:8000/

# Upload documents (example)
curl -X POST "http://localhost:8000/feed" \
  -F "files=@your_document.pdf" \
  -F "filepaths=your_document.pdf" \
  -F "collection_id=test-collection"
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs for error messages
3. Ensure all dependencies are properly installed
4. Verify your API keys and model configurations
