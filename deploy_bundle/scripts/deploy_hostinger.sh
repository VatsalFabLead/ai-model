#!/usr/bin/env bash
# Safe deploy to ONE Hostinger folder only (fabai.fableadtech.in).
# Does NOT delete remote files. Does NOT touch other domains/projects.
#
# Git Bash usage:
#   export FTP_HOST="82.29.163.188"
#   export FTP_USER="u378554361.fabaifptusr"
#   export FTP_PASSWORD="your-ftp-password"
#   bash scripts/deploy_hostinger.sh
#
# Or PowerShell:
#   powershell.exe -ExecutionPolicy Bypass -File scripts/deploy_hostinger.ps1

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FTP_HOST="${FTP_HOST:-}"
FTP_USER="${FTP_USER:-}"
FTP_PASSWORD="${FTP_PASSWORD:-}"
FTP_SERVER_DIR="${FTP_SERVER_DIR:-/home/u378554361/domains/fabai.fableadtech.in/public_html}"
FTP_PORT="${FTP_PORT:-65002}"
BUNDLE_DIR="deploy_bundle"

# Safety: only allow deploy inside this domain folder (never / or other sites).
ALLOWED_MARKER="domains/fabai.fableadtech.in/public_html"
if [[ "$FTP_SERVER_DIR" != *"$ALLOWED_MARKER"* ]]; then
  echo "REFUSED: Remote path must contain: $ALLOWED_MARKER"
  echo "Got: $FTP_SERVER_DIR"
  echo "This protects your other live projects on the server."
  exit 1
fi

if [[ -z "$FTP_HOST" || -z "$FTP_USER" ]]; then
  echo "Set FTP_HOST and FTP_USER first."
  echo '  export FTP_HOST="82.29.163.188"'
  echo '  export FTP_USER="u378554361.fabaifptusr"'
  echo '  export FTP_PASSWORD="your-password"   # optional if you type at prompt'
  exit 1
fi

echo "=== Safe deploy (fabai only) ==="
echo "Local:  $ROOT"
echo "Remote: $FTP_SERVER_DIR"
echo "       (other domains on your account are NOT touched)"
echo ""

rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR"

# Copy project files locally — excludes secrets and junk.
shopt -s dotglob
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
shopt -u dotglob

find "$BUNDLE_DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo "Bundle built. Uploading (add/overwrite files only, no remote delete)..."

REMOTE="${FTP_USER}@${FTP_HOST}:${FTP_SERVER_DIR}/"

if [[ -n "$FTP_PASSWORD" ]] && command -v sshpass >/dev/null 2>&1; then
  sshpass -p "$FTP_PASSWORD" scp -P "$FTP_PORT" -o StrictHostKeyChecking=no -r "$BUNDLE_DIR"/* "$REMOTE"
elif [[ -n "$FTP_PASSWORD" ]]; then
  echo "Tip: install sshpass for non-interactive upload, or enter password when prompted."
  scp -P "$FTP_PORT" -o StrictHostKeyChecking=no -r "$BUNDLE_DIR"/* "$REMOTE"
else
  echo "Enter FTP password when prompted:"
  scp -P "$FTP_PORT" -o StrictHostKeyChecking=no -r "$BUNDLE_DIR"/* "$REMOTE"
fi

echo ""
echo "Done. Files uploaded ONLY to: $FTP_SERVER_DIR"
echo ""
echo "Next — SSH (commands affect ONLY this folder):"
echo "  cd ~/domains/fabai.fableadtech.in/public_html"
echo "  source .venv/bin/activate"
echo "  pip install -r requirements.txt"
echo ""
echo "Restart ONLY this app (hPanel Python Restart is safest if you have multiple sites):"
echo "  cd ~/domains/fabai.fableadtech.in/public_html"
echo "  pkill -f 'domains/fabai.fableadtech.in/public_html.*gunicorn' 2>/dev/null || true"
echo "  nohup gunicorn app.main:app -c gunicorn.conf.py > app.log 2>&1 &"
