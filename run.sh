#!/bin/bash

# Uvicorn FastAPI Server Configuration
# Read configuration from environment variables with sensible defaults

# Core application parameters
APP="app.api.main:app"  # The FastAPI application module and instance

# Network parameters - use environment variables with Docker-friendly defaults
HOST="${HOST:-0.0.0.0}"  # Use environment variable, default to 0.0.0.0 for Docker
PORT="${FASTAPI_PORT:-8000}"  # Use FASTAPI_PORT from environment, default to 8000

# Performance parameters
WORKERS="${WORKERS:-8}"       # Number of worker processes
LOOP="${LOOP:-uvloop}"        # Event loop implementation
HTTP="${HTTP:-httptools}"     # HTTP protocol implementation

# Timeout parameters
TIMEOUT_KEEP_ALIVE="${TIMEOUT_KEEP_ALIVE:-5}"           # Keep-alive timeout in seconds
TIMEOUT_GRACEFUL_SHUTDOWN="${TIMEOUT_GRACEFUL_SHUTDOWN:-30}"   # Graceful shutdown timeout in seconds

# Concurrency parameters
LIMIT_CONCURRENCY="${LIMIT_CONCURRENCY:-1000}"         # Maximum concurrent connections

# Logging parameters
LOG_LEVEL="${LOG_LEVEL:-info}"  # Log level (debug, info, warning, error, critical)
ACCESS_LOG="${ACCESS_LOG:-true}"  # Enable access logging


# Export environment variables for use in Python code
export UVICORN_HOST="$HOST"
export UVICORN_PORT="$PORT"
export UVICORN_WORKERS="$WORKERS"
export UVICORN_LOOP="$LOOP"
export UVICORN_HTTP="$HTTP"
export UVICORN_TIMEOUT_KEEP_ALIVE="$TIMEOUT_KEEP_ALIVE"
export UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN="$TIMEOUT_GRACEFUL_SHUTDOWN"
export UVICORN_LIMIT_CONCURRENCY="$LIMIT_CONCURRENCY"

echo "Starting FastAPI server with uvicorn..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"
echo "Loop: $LOOP"
echo "HTTP: $HTTP"
echo "Timeout Keep Alive: ${TIMEOUT_KEEP_ALIVE}s"
echo "Timeout Graceful Shutdown: ${TIMEOUT_GRACEFUL_SHUTDOWN}s"
echo "Limit Concurrency: $LIMIT_CONCURRENCY"
echo "Log Level: $LOG_LEVEL"
echo "Access Log: $ACCESS_LOG"
echo "----------------------------------------"

# Error handling function
handle_error() {
    echo "Error: FastAPI server failed to start or crashed"
    echo "Exit code: $?"
    exit 1
}

# Set up error handling
trap handle_error ERR

# Run uvicorn with all parameters
uvicorn "$APP" \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --loop "$LOOP" \
        --http "$HTTP" \
        --timeout-keep-alive "$TIMEOUT_KEEP_ALIVE" \
        --timeout-graceful-shutdown "$TIMEOUT_GRACEFUL_SHUTDOWN" \
        --limit-concurrency "$LIMIT_CONCURRENCY" \
        --log-level "$LOG_LEVEL" \
        --access-log