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

if [ ! -f "$ROOT/backend/.env" ]; then
    echo "✗  backend/.env not found. Create it first:"
    echo ""
    echo "    cat > backend/.env << EOF"
    echo "    MONGO_URL=mongodb://localhost:27017"
    echo "    DB_NAME=quantfi"
    echo "    NEWS_API_KEY=your_key"
    echo "    EMERGENT_LLM_KEY=your_key"
    echo "    CORS_ORIGINS=http://localhost:$FRONTEND_PORT"
    echo "    OPENAI_API_KEY=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" 
    echo "    EOF"
    echo ""
    exit 1
fi

cd "$ROOT/backend"
if [ ! -d ".venv" ]; then
    echo "→ Creating backend venv..."
    python3 -m venv .venv
fi
CORS_EFFECTIVE="${CORS_ORIGINS:-http://localhost:$FRONTEND_PORT}"
echo "→ Syncing backend deps..."
PIP_DISABLE_PIP_VERSION_CHECK=1 .venv/bin/pip install -q -r requirements.txt
echo "→ Backend starting on :$BACKEND_PORT"
CORS_ORIGINS="$CORS_EFFECTIVE" .venv/bin/uvicorn server:app --host 0.0.0.0 --port "$BACKEND_PORT" &
BACKEND_PID=$!

cd "$ROOT/frontend"
if [ ! -d "node_modules" ]; then
    echo "→ Installing frontend deps..."
    yarn install --frozen-lockfile
fi
echo "→ Frontend starting on :$FRONTEND_PORT"
REACT_APP_BACKEND_URL="http://localhost:$BACKEND_PORT" PORT="$FRONTEND_PORT" yarn start &
FRONTEND_PID=$!

echo ""
echo "  backend  → http://localhost:$BACKEND_PORT"
echo "  frontend → http://localhost:$FRONTEND_PORT"
echo "  Ctrl+C to stop both"
echo ""

wait
