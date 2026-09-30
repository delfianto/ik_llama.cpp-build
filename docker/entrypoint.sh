#!/bin/bash

# Default values
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}
CTX_SIZE=${CTX_SIZE:-8192}
N_GPU_LAYERS=${N_GPU_LAYERS:--1}
THREADS=${THREADS:-$(nproc)}

# Base command
CMD=("/usr/local/bin/llama-server" "--host" "${HOST}" "--port" "${PORT}" "-c" "${CTX_SIZE}" "-ngl" "${N_GPU_LAYERS}" "-t" "${THREADS}")

# Check for main model
if [ -n "$MODEL_PATH" ]; then
    CMD+=("-m" "$MODEL_PATH")
else
    echo "ERROR: MODEL_PATH environment variable is required."
    exit 1
fi

# Check for an external MTP companion model.
if [ -n "$MTP_MODEL_PATH" ]; then
    MTP_DRAFT_N=${MTP_DRAFT_N:-3}
    echo "MTP companion detected. Enabling Multi-Token Prediction at depth ${MTP_DRAFT_N}..."
    CMD+=("--model-draft" "$MTP_MODEL_PATH")
    CMD+=("--spec-type" "mtp:n_max=${MTP_DRAFT_N},p_min=0.0")
fi

# Add any additional user-provided arguments
if [ -n "$EXTRA_ARGS" ]; then
    # We deliberately do not quote EXTRA_ARGS here so it splits into separate tokens
    for arg in $EXTRA_ARGS; do
        CMD+=("$arg")
    done
fi

echo "Starting llama-server with command: ${CMD[*]}"
exec "${CMD[@]}"
