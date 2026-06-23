# Title & Meta Generator test
param(
  [string]$BaseUrl = "http://127.0.0.1:8000",
  [string]$ApiKey = $env:API_KEY,
  [int]$TimeoutSec = 90
)

if (-not $ApiKey) { throw "Set API_KEY or pass -ApiKey" }
$Headers = @{ "Authorization" = "Bearer $ApiKey"; "Content-Type" = "application/json" }

$body = @{
  topic       = "Email Marketing Best Practices"
  variations  = 10
  tone        = "professional"
  category    = "blog_article"
  language    = "English"
  use_ai      = $false
  use_rag     = $true
} | ConvertTo-Json -Compress

Write-Host "GET $BaseUrl/v1/title-meta/version" -ForegroundColor Yellow
$ver = Invoke-RestMethod -Uri "$BaseUrl/v1/title-meta/version" -Headers $Headers -TimeoutSec 30
Write-Host ("Version: {0}" -f $ver.generator_version) -ForegroundColor Cyan

Write-Host "POST $BaseUrl/v1/title-meta/generate" -ForegroundColor Yellow
$r = Invoke-RestMethod -Uri "$BaseUrl/v1/title-meta/generate" -Method Post -Headers $Headers -Body $body -TimeoutSec $TimeoutSec

Write-Host ("Topic: {0} | Variations: {1} | Version: {2} | Seed: {3}" -f $r.topic, $r.variation_count, $r.generator_version, $r.variation_seed) -ForegroundColor Green
$r.variations | Select-Object title, title_length, meta_length, quality_score, seo_ready | Format-Table -AutoSize
Write-Host "`n--- Variation 1 ---"
Write-Host "Title: $($r.variations[0].title)"
Write-Host "Meta:  $($r.variations[0].meta_description)"
