# SEO Optimizer test - matches UI metrics
param(
  [string]$BaseUrl = "http://127.0.0.1:8000",
  [string]$ApiKey = $env:API_KEY,
  [int]$TimeoutSec = 120
)

$Headers = @{ "Authorization" = "Bearer $ApiKey"; "Content-Type" = "application/json" }

$body = @{
  content  = "email marketing is good. many people use email marketing for business. email marketing helps you reach customers. you should try email marketing today because email marketing is very important for growth."
  keywords = "email marketing"
  tone     = "professional"
  category = "blog_article"
  use_ai   = $true
} | ConvertTo-Json -Compress

Write-Host "POST $BaseUrl/v1/seo-optimizer/optimize" -ForegroundColor Yellow
$r = Invoke-RestMethod -Uri "$BaseUrl/v1/seo-optimizer/optimize" -Method Post -Headers $Headers -Body $body -TimeoutSec $TimeoutSec

Write-Host "`n--- BEFORE ---" -ForegroundColor Cyan
$r.original | Format-List readability_score, word_count, character_count, sentence_count
Write-Host "SEO score: $($r.seo_score_before)"

Write-Host "`n--- AFTER ---" -ForegroundColor Green
$r.optimized | Format-List readability_score, word_count, character_count, sentence_count
Write-Host "SEO score: $($r.seo_score_after) (improvement: +$($r.improvement))"

Write-Host "`n--- Suggestions ---"
$r.suggestions | ForEach-Object { Write-Host "  - $_" }

Write-Host "`n--- Optimized Content ---"
Write-Host $r.optimized_content
