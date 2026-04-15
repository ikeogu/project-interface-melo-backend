#!/bin/bash
# Start the LiveKit agent worker
# Run this alongside the FastAPI server

cd "$(dirname "$0")/.."

echo "Starting AI Contacts LiveKit Agent..."
echo "Connecting to: ${LIVEKIT_URL:-ws://localhost:7880}"
echo ""

pip install -r agent/requirements.txt -q

python -m agent.main start
