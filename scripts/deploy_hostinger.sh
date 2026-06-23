#!/usr/bin/env bash
# Safe deploy to fabai.fableadtech.in/public_html ONLY.
# Git Bash: run export lines first, then: bash scripts/deploy_hostinger.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SAFE_REMOTE="/home/u378554361/domains/fabai.fableadtech.in/public_html"
FTP_HOST="${FTP_HOST:-}"
FTP_USER="${FTP_USER:-}"
FTP_PASSWORD="${FTP_PASSWORD:-}"
FTP_SERVER_DIR="${FTP_SERVER_DIR:-$SAFE_REMOTE}"
FTP_PORT="${FTP_PORT:-65002}"
BUNDLE_DIR="deploy_bundle"

# Git Bash turns /home/... into C:/Program Files/Git/home/... — fix that.
FTP_SERVER_DIR="${FTP_SERVER_DIR//\\//}"
FTP_SERVER_DIR="$(echo "$FTP_SERVER_DIR" | sed 's|^//|/|' | sed 's|C:/Program Files/Git||' | sed 's|^/||')"
if [[ "$FTP_SERVER_DIR" != /* ]]; then
  FTP_SERVER_DIR="/${FTP_SERVER_DIR}"
fi
FTP_SERVER_DIR="${FTP_SERVER_DIR%/}"

if [[ "$FTP_SERVER_DIR" != "$SAFE_REMOTE" ]]; then
  echo "REFUSED: remote path must be exactly:"
  echo "  $SAFE_REMOTE"
  echo "Got: $FTP_SERVER_DIR"
  echo "(protects your other live projects on the server)"
  exit 1
fi

if [[ -z "$FTP_HOST" || -z "$FTP_USER" ]]; then
  echo "Set variables first (Git Bash syntax):"
  echo '  export FTP_HOST="82.29.163.188"'
  echo '  export FTP_USER="u378554361"'
  echo '  export FTP_SERVER_DIR="/home/u378554361/domains/fabai.fableadtech.in/public_html"'
  exit 1
fi

echo "=== Safe deploy (fabai only) ==="
echo "Local:  $ROOT"
echo "Remote: $FTP_SERVER_DIR"
echo ""

rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR"

shopt -s dotglob nullglob
for item in *; do
  case "$item" in
    .git|.github|.venv|venv|__pycache__|.pytest_cache|.mypy_cache|node_modules|deploy_bundle)
      continue ;;
    .env|.env.*)
      continue ;;
    my_resume.md|resume_output.md)
      continue ;;
  esac
  cp -a "$item" "$BUNDLE_DIR/"
done
shopt -u dotglob nullglob

find "$BUNDLE_DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

if [[ -z "$(ls -A "$BUNDLE_DIR" 2>/dev/null)" ]]; then
  echo "ERROR: deploy bundle is empty."
  exit 1
fi

echo "Bundle ready ($(ls -1 "$BUNDLE_DIR" | wc -l) items). Uploading..."
echo "Enter SSH password when prompted."

REMOTE="${FTP_USER}@${FTP_HOST}:${FTP_SERVER_DIR}/"
export MSYS_NO_PATHCONV=1

if [[ -n "$FTP_PASSWORD" ]] && command -v sshpass >/dev/null 2>&1; then
  sshpass -p "$FTP_PASSWORD" scp -P "$FTP_PORT" -o StrictHostKeyChecking=no -r "$BUNDLE_DIR/." "$REMOTE"
else
  scp -P "$FTP_PORT" -o StrictHostKeyChecking=no -r "$BUNDLE_DIR/." "$REMOTE"
fi

echo ""
echo "Done. Uploaded ONLY to fabai.fableadtech.in/public_html"
echo "SSH next:"
echo "  cd ~/domains/fabai.fableadtech.in/public_html"
echo "  source .venv/bin/activate && pip install -r requirements.txt"
