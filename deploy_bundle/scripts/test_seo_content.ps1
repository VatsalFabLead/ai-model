# SEO Content Generator - single API test
# Usage: .\scripts\test_seo_content.ps1

param(
  [string]$BaseUrl = "http://127.0.0.1:8000",
  [string]$ApiKey = $env:API_KEY,
  [int]$TimeoutSec = 120
)

$Headers = @{ "Authorization" = "Bearer $ApiKey"; "Content-Type" = "application/json" }

Write-Host "POST $BaseUrl/v1/seo-content/generate" -ForegroundColor Yellow

$body = @{
  topic             = "Email marketing best practices"
  keywords          = "email marketing, newsletters, conversions"
  tone              = "professional"
  word_count        = 400
  category          = "blog_article"
  language          = "English"
  audience          = "small business owners"
  use_ai            = $true
  discover_keywords = $false
} | ConvertTo-Json -Compress

$r = Invoke-RestMethod -Uri "$BaseUrl/v1/seo-content/generate" -Method Post -Headers $Headers -Body $body -TimeoutSec $TimeoutSec

Write-Host ("OK: {0} | SEO score: {1}% | Words: {2}" -f $r.title, $r.quality.seo_score, $r.word_count) -ForegroundColor Green
Write-Host ("Category: {0} | Lang: {1} | AI used: {2}" -f $r.category, $r.language, $r.ai.model_used)
Write-Host ("Meta: {0}" -f $r.meta_description)
Write-Host ("Keywords: {0}" -f ($r.keywords -join ', '))
Write-Host "`n--- Content preview ---"
$r.content.Substring(0, [Math]::Min(800, $r.content.Length))
if ($r.content.Length -gt 800) { Write-Host "..." }
