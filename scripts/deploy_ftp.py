"""
Automated FTP Deployment script for Hostinger shared hosting (fabai.fableadtech.in).
"""

import os
import sys
import ftplib
from pathlib import Path

raw_host = os.environ.get("FTP_HOST", "").strip() or "82.29.163.188"
for prefix in ("ftp://", "ftps://", "http://", "https://"):
    if raw_host.startswith(prefix):
        raw_host = raw_host[len(prefix):]
FTP_HOST = raw_host.split("/")[0].split(":")[0].strip() or "82.29.163.188"

FTP_USER = os.environ.get("FTP_USER", "").strip() or "u378554361.fabaifptusr"
FTP_PASS = os.environ.get("FTP_PASS", "").strip() or "W;0xUPg>3XL"
FTP_PORT = int(os.environ.get("FTP_PORT", 21) or 21)

EXCLUDE_DIRS = {
    ".git",
    ".github",
    ".venv",
    ".venv-py314-backup",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "deploy_bundle",
}

EXCLUDE_FILES = {
    ".ftp-deploy-sync-state.json",
}

ROOT_DIR = Path(__file__).resolve().parent.parent

def connect_ftp():
    print(f"Connecting to FTP server {FTP_HOST}:{FTP_PORT} as {FTP_USER}...")
    ftp = ftplib.FTP()
    ftp.connect(FTP_HOST, FTP_PORT, timeout=30)
    ftp.login(FTP_USER, FTP_PASS)
    print("FTP Login successful!")
    return ftp

def remove_remote_default_php(ftp):
    try:
        remote_files = ftp.nlst()
        if "default.php" in remote_files:
            print("Removing default Hostinger page (default.php)...")
            ftp.delete("default.php")
            print("default.php removed successfully!")
    except Exception as e:
        print(f"Note on default.php check: {e}")

def ensure_remote_dir(ftp, remote_dir_path):
    # Ensure all ancestor directories exist starting from FTP root '/'
    parts = [p for p in remote_dir_path.replace("\\", "/").split("/") if p]
    current = ""
    for part in parts:
        current += "/" + part
        try:
            ftp.cwd(current)
        except ftplib.error_perm:
            try:
                ftp.mkd(current)
                print(f"Created remote directory: {current}")
            except Exception as e:
                print(f"Error creating directory {current}: {e}")

def upload_file(ftp, local_path, remote_path):
    remote_path_clean = remote_path.replace("\\", "/")
    remote_dir = os.path.dirname(remote_path_clean)
    
    if remote_dir:
        ensure_remote_dir(ftp, remote_dir)
    else:
        ftp.cwd("/")
        
    filename = os.path.basename(local_path)
    
    with open(local_path, "rb") as f:
        ftp.storbinary(f"STOR {filename}", f)
    print(f"Uploaded: /{remote_path_clean}")

def deploy():
    ftp = connect_ftp()
    remove_remote_default_php(ftp)
    
    print("\nStarting FTP file upload...")
    uploaded_count = 0
    
    for root, dirs, files in os.walk(ROOT_DIR):
        # Prune excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
        
        rel_root = os.path.relpath(root, ROOT_DIR)
        if rel_root == ".":
            rel_root = ""
            
        for file in files:
            if file in EXCLUDE_FILES or file.endswith(".pyc") or file.endswith(".pyo"):
                continue
                
            local_file_path = os.path.join(root, file)
            remote_file_path = os.path.join(rel_root, file).replace("\\", "/")
            
            upload_file(ftp, local_file_path, remote_file_path)
            uploaded_count += 1
            
    print(f"\nFTP deployment finished! Total files uploaded: {uploaded_count}")
    
    # Touch tmp/restart.txt for Phusion Passenger restart
    try:
        ensure_remote_dir(ftp, "tmp")
        import io
        ftp.cwd("/tmp")
        ftp.storbinary("STOR restart.txt", io.BytesIO(b"restart"))
        print("Touched /tmp/restart.txt to restart application server.")
    except Exception as e:
        print(f"Could not touch tmp/restart.txt: {e}")

    try:
        ftp.quit()
    except Exception:
        pass

if __name__ == "__main__":
    deploy()
