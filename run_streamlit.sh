#!/bin/bash

# Streamlit Frontend Configuration
# Read configuration from environment variables with sensible defaults

# Streamlit server parameters - use environment variables with Docker-friendly defaults
STREAMLIT_HOST="${STREAMLIT_HOST:-0.0.0.0}"        # Server address (0.0.0.0 for external access)
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"           # Server port
STREAMLIT_APP="${STREAMLIT_APP:-app.py}"

# API connection parameters (needed by app.py)
HOST="${HOST:-fastapi}"                            # FastAPI host (default to 'fastapi' for Docker)
FASTAPI_PORT="${FASTAPI_PORT:-8000}" 

# Streamlit environment variables
export STREAMLIT_BROWSER_GATHERUSAGESTATS="${STREAMLIT_BROWSER_GATHERUSAGESTATS:-false}"
export STREAMLIT_SERVER_HEADLESS="${STREAMLIT_SERVER_HEADLESS:-true}"
export STREAMLIT_SERVER_ENABLE_CORS="${STREAMLIT_SERVER_ENABLE_CORS:-false}"
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION="${STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION:-true}"

# Export API connection variables for app.py
export HOST="$HOST"
export FASTAPI_PORT="$FASTAPI_PORT"

echo "Starting Streamlit frontend..."
echo "Host: $STREAMLIT_HOST"
echo "Port: $STREAMLIT_PORT"
echo "App: $STREAMLIT_APP"
echo "API Host: $HOST"
echo "API Port: $FASTAPI_PORT"
echo "Browser Stats: $STREAMLIT_BROWSER_GATHERUSAGESTATS"
echo "Headless: $STREAMLIT_SERVER_HEADLESS"
echo "CORS: $STREAMLIT_SERVER_ENABLE_CORS"
echo "XSRF Protection: $STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"
echo "----------------------------------------"

# Error handling function
handle_error() {
    echo "Error: Streamlit server failed to start or crashed"
    echo "Exit code: $?"
    exit 1
}

# Set up error handling
trap handle_error ERR


# Run streamlit
streamlit run "$STREAMLIT_APP" \
    --server.address "$STREAMLIT_HOST" \
    --server.port "$STREAMLIT_PORT" \
    --server.enableCORS "$STREAMLIT_SERVER_ENABLE_CORS" \
    --server.enableXsrfProtection "$STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"