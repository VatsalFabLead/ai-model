# Deploy to Hostinger shared hosting from your PC (SFTP port 65002).
# Uploads ONLY to fabai.fableadtech.in/public_html - other projects are not touched.
#
# Usage:
#   $env:FTP_HOST = "82.29.163.188"
#   $env:FTP_USER = "u378554361.fabaifptusr"
#   $env:FTP_SERVER_DIR = "/home/u378554361/domains/fabai.fableadtech.in/public_html"
#   powershell -ExecutionPolicy Bypass -File scripts\deploy_hostinger.ps1

param(
  [string]$FtpHost = $env:FTP_HOST,
  [string]$User = $env:FTP_USER,
  [string]$RemoteDir = $env:FTP_SERVER_DIR,
  [int]$Port = 65002,
  [string]$BundleDir = "deploy_bundle"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

if (-not $FtpHost -or -not $User -or -not $RemoteDir) {
  Write-Host "Set FTP_HOST, FTP_USER, FTP_SERVER_DIR first." -ForegroundColor Red
  exit 1
}

$allowed = "domains/fabai.fableadtech.in/public_html"
if ($RemoteDir -notlike "*$allowed*") {
  Write-Host "REFUSED: path must contain $allowed (protects other live sites)." -ForegroundColor Red
  exit 1
}

Write-Host "Safe deploy - ONLY to: $RemoteDir" -ForegroundColor Green
Write-Host "Building deploy bundle..." -ForegroundColor Cyan
if (Test-Path $BundleDir) { Remove-Item $BundleDir -Recurse -Force }
New-Item -ItemType Directory -Path $BundleDir | Out-Null

$excludeDirs = @(
  ".git", ".github", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
  "node_modules", "deploy_bundle", "models\llm"
)
$excludeFiles = @(".env", ".env.example")

Get-ChildItem -Force | Where-Object {
  $_.Name -notin $excludeDirs -and $_.Name -notin $excludeFiles
} | ForEach-Object {
  Copy-Item $_.FullName -Destination $BundleDir -Recurse -Force
}

Get-ChildItem $BundleDir -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Uploading via SCP port $Port (enter FTP password when prompted)..." -ForegroundColor Cyan

$bundlePath = (Resolve-Path $BundleDir).Path
$remote = "${User}@${FtpHost}:${RemoteDir}/"

try {
  & scp -P $Port -o StrictHostKeyChecking=no -r "$bundlePath\*" $remote
  if ($LASTEXITCODE -ne 0) { throw "scp failed with exit code $LASTEXITCODE" }
  Write-Host ""
  Write-Host "Deploy complete." -ForegroundColor Green
  Write-Host "SSH (fabai folder only):" -ForegroundColor Cyan
  Write-Host "  cd ~/domains/fabai.fableadtech.in/public_html"
  Write-Host "  source .venv/bin/activate"
  Write-Host "  pip install -r requirements.txt"
}
catch {
  Write-Host ""
  Write-Host "Deploy failed: $_" -ForegroundColor Red
  Write-Host "Use FileZilla: SFTP port 65002, folder fabai.fableadtech.in/public_html only" -ForegroundColor Yellow
  exit 1
}
