#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

find_free_port() {
    local preferred=$1
    if ! lsof -i :"$preferred" -t >/dev/null 2>&1; then
        echo "$preferred"
        return
    fi
    echo >&2 "⚠  Port $preferred in use, scanning for free port..."
    local port=$((preferred + 1))
    while lsof -i :"$port" -t >/dev/null 2>&1; do
        port=$((port + 1))
        if [ "$port" -gt "$((preferred + 100))" ]; then
            echo >&2 "✗  No free port found in range $preferred–$port"
            exit 1
        fi
    done
    echo "$port"
}

BACKEND_PORT=$(find_free_port "${BACKEND_PORT:-8000}")
FRONTEND_PORT=$(find_free_port "${PORT:-3000}")

cleanup() {
    echo ""
    echo "Shutting down..."
    kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
    wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
    exit 0
}
trap cleanup INT TERM

if [ -f "$ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$ROOT/.env"
    set +a
elif [ -f "$ROOT/backend/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$ROOT/backend/.env"
    set +a
else
    echo "✗  .env not found. Copy the example and fill in your keys:"
    echo ""
    echo "    cp .env.example .env"
    echo ""
    exit 1
fi

cd "$ROOT/backend"
if [ ! -d ".venv" ]; then
    echo "→ Creating backend venv..."
    python3 -m venv .venv
fi
CORS_EFFECTIVE="http://localhost:$FRONTEND_PORT"
echo "→ Syncing backend deps..."
PIP_DISABLE_PIP_VERSION_CHECK=1 .venv/bin/pip install -q -r requirements.txt
if [ -f "$ROOT/requirements-bot.txt" ]; then
    echo "→ Syncing bot deps..."
    PIP_DISABLE_PIP_VERSION_CHECK=1 .venv/bin/pip install -q -r "$ROOT/requirements-bot.txt"
fi
echo "→ Backend starting on :$BACKEND_PORT"
cd "$ROOT"
CORS_ORIGINS="$CORS_EFFECTIVE" PYTHONPATH="$ROOT" backend/.venv/bin/uvicorn backend.server:app --host 0.0.0.0 --port "$BACKEND_PORT" &
BACKEND_PID=$!

cd "$ROOT/frontend"
if [ ! -d "node_modules" ]; then
    echo "→ Installing frontend deps..."
    npm install
fi
echo "→ Frontend starting on :$FRONTEND_PORT"
VITE_BACKEND_URL="http://localhost:$BACKEND_PORT" npx vite --port "$FRONTEND_PORT" &
FRONTEND_PID=$!

echo ""
echo "  backend  → http://localhost:$BACKEND_PORT"
echo "  frontend → http://localhost:$FRONTEND_PORT"
echo "  Ctrl+C to stop both"
echo ""

wait
