#!/bin/bash

# Uvicorn FastAPI Server Configuration
# Default values for uvicorn parameters

# Core application parameters
APP="api:app"  # The FastAPI application module and instance

# Network parameters
HOST="127.0.0.1"  # Default host (localhost)
PORT="8000"       # Default port

# Performance parameters
WORKERS="1"       # Number of worker processes (default: 1)
LOOP="uvloop"      # Event loop implementation (auto, asyncio, uvloop)
HTTP="httptools"   # HTTP protocol implementation (auto, h11, httptools)

# Timeout parameters
TIMEOUT_KEEP_ALIVE="5"           # Keep-alive timeout in seconds (default: 5)
TIMEOUT_GRACEFUL_SHUTDOWN="30"   # Graceful shutdown timeout in seconds (default: 30)

# Concurrency parameters
LIMIT_CONCURRENCY="1000"         # Maximum concurrent connections (default: 1000)


# Export environment variables (optional - for use in Python code)
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
echo "Reload: $RELOAD"
echo "----------------------------------------"

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
    --access-log \
    --no-reload
