# Test all Custom Model API endpoints from PowerShell
# Usage:
#   .\scripts\test_all_apis.ps1
#   .\scripts\test_all_apis.ps1 -BaseUrl "https://ai-model-api-2906.onrender.com" -ApiKey "YOUR_KEY"

param(
  [string]$BaseUrl = "http://127.0.0.1:8000",
  [string]$ApiKey = $env:API_KEY,
  [int]$TimeoutSec = 180
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

if (-not $ApiKey) {
  throw "Set API_KEY environment variable or pass -ApiKey"
}

$Headers = @{
  "Authorization" = "Bearer $ApiKey"
  "Content-Type"  = "application/json"
}

function Test-Get {
  param([string]$Name, [string]$Path, [switch]$NoAuth)
  Write-Host "`n=== $Name ===" -ForegroundColor Cyan
  try {
    if ($NoAuth) {
      $r = Invoke-RestMethod -Uri "$BaseUrl$Path" -TimeoutSec 30
    } else {
      $r = Invoke-RestMethod -Uri "$BaseUrl$Path" -Headers $Headers -TimeoutSec 30
    }
    Write-Host "OK" -ForegroundColor Green
    $r | ConvertTo-Json -Depth 4 -Compress | ForEach-Object { $_.Substring(0, [Math]::Min(500, $_.Length)) }
    if ($_.Length -gt 500) { Write-Host "..." }
    return $true
  } catch {
    Write-Host "FAIL: $($_.Exception.Message)" -ForegroundColor Red
    return $false
  }
}

function Test-Post {
  param([string]$Name, [string]$Path, [object]$Body)
  Write-Host "`n=== $Name ===" -ForegroundColor Cyan
  try {
    $json = $Body | ConvertTo-Json -Depth 10 -Compress
    $r = Invoke-RestMethod -Uri "$BaseUrl$Path" -Method Post -Headers $Headers -Body $json -TimeoutSec $TimeoutSec
    Write-Host "OK" -ForegroundColor Green
    $out = $r | ConvertTo-Json -Depth 6 -Compress
    if ($out.Length -gt 600) { Write-Host ($out.Substring(0, 600) + "...") } else { Write-Host $out }
    return $true
  } catch {
    Write-Host "FAIL: $($_.Exception.Message)" -ForegroundColor Red
    return $false
  }
}

Write-Host "Base URL: $BaseUrl" -ForegroundColor Yellow
Write-Host "Timeout: ${TimeoutSec}s per POST" -ForegroundColor Yellow

$ok = 0
$fail = 0

if (Test-Get "Health" "/health" -NoAuth) { $ok++ } else { $fail++ }
if (Test-Get "Root" "/" -NoAuth) { $ok++ } else { $fail++ }
if (Test-Get "Models" "/v1/models") { $ok++ } else { $fail++ }

if (Test-Post "Chat" "/v1/chat/completions" @{
    model = "custom-nexus-v1"
    messages = @(@{ role = "user"; content = "Say hello in one sentence." })
    max_tokens = 60
  }) { $ok++ } else { $fail++ }

if (Test-Get "Post Scheduler - Platforms" "/v1/post-scheduler/platforms") { $ok++ } else { $fail++ }
if (Test-Post "Post Scheduler - Content" "/v1/post-scheduler/suggest-content" @{
    platform = "linkedin"; topic = "Product launch tips"; tone = "professional"; include_emojis = $false
  }) { $ok++ } else { $fail++ }
if (Test-Post "Post Scheduler - Hashtags" "/v1/post-scheduler/suggest-hashtags" @{
    platform = "instagram"; topic = "digital marketing"; count = 5
  }) { $ok++ } else { $fail++ }

if (Test-Post "SEO Content" "/v1/seo-content/generate" @{
    topic = "Email marketing basics"; keywords = "email, marketing"; tone = "professional"
    word_count = 300; category = "blog_article"; language = "English"; use_ai = $true
  }) { $ok++ } else { $fail++ }

if (Test-Post "Title & Meta" "/v1/title-meta/generate" @{
    topic = "Email Marketing Best Practices"; variations = 2; tone = "professional"
    category = "blog_article"; language = "English"; use_ai = $true
  }) { $ok++ } else { $fail++ }

if (Test-Get "Schema Markup - Types" "/v1/schema-markup/types") { $ok++ } else { $fail++ }
if (Test-Get "Schema Markup - Categories" "/v1/schema-markup/categories") { $ok++ } else { $fail++ }
if (Test-Get "Schema Markup - Languages" "/v1/schema-markup/languages") { $ok++ } else { $fail++ }
if (Test-Post "Schema Markup - Generate" "/v1/schema-markup/generate" @{
    schema_type = "Article"; name = "Email Marketing Guide"; language = "English"
    data = @{ description = "A practical guide"; url = "https://example.com" }; ai_enhance = $false
  }) { $ok++ } else { $fail++ }

if (Test-Post "SEO Keywords" "/v1/seo-keywords/generate" @{
    seed_keyword = "digital marketing"; max_items = 8; use_ai = $true; discover_web = $true
  }) { $ok++ } else { $fail++ }

if (Test-Post "Email - New" "/v1/email-assistant/new-email" @{
    subject = "Project update"; context = "Inform client about 3-day delay, new ETA Friday."; tone = "professional"
  }) { $ok++ } else { $fail++ }
if (Test-Post "Email - Reply" "/v1/email-assistant/reply" @{
    original_email = "Can we still release Tuesday?"; reply_points = "Thank them, QA issue, new ETA Friday."; tone = "professional"
  }) { $ok++ } else { $fail++ }
if (Test-Post "Email - Cold" "/v1/email-assistant/cold-email" @{
    company_name = "Acme Corp"; purpose_offer = "AI support automation"; value_proposition = "Faster responses, better CSAT"; tone = "professional"
  }) { $ok++ } else { $fail++ }

if (Test-Post "Resume - Generate" "/v1/resume-builder/generate" @{
    full_name = "Test User"; job_title = "Flutter Developer"
    email = "test@example.com"; phone = "+91-9999999999"
    linkedin = "https://linkedin.com/in/testuser"
    portfolio = "https://github.com/testuser"
    education = "B.Tech CS — GTU, 2022"
    experience = "Built Flutter apps, integrated Firebase"
    projects = "- E-Commerce App"; certifications = "- Flutter Cert"
    achievements = "- Hackathon Winner"; languages = "English, Hindi"
    template = "modern"; language = "English"; use_ai = $true
  }) { $ok++ } else { $fail++ }

if (Test-Post "Cover Letter" "/v1/cover-letter/generate" @{
    job_role = "Software Engineer"; company_name = "Microsoft"
    skills_experience = "3 years Python, built APIs, team collaboration."; tone = "professional"
  }) { $ok++ } else { $fail++ }

Write-Host "`n========================================" -ForegroundColor Yellow
Write-Host "PASSED: $ok  |  FAILED: $fail" -ForegroundColor $(if ($fail -eq 0) { "Green" } else { "Yellow" })
Write-Host "Docs: $BaseUrl/docs" -ForegroundColor Gray
