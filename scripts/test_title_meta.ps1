# Title & Meta Generator test
param(
  [string]$BaseUrl = "http://127.0.0.1:8000",
  [string]$ApiKey = "wYGt5Pq-fUUVzO9mWVtkhoVwLWBWPjEY8InlnWqoqDshkVl2",
  [int]$TimeoutSec = 90
)

$Headers = @{ "Authorization" = "Bearer $ApiKey"; "Content-Type" = "application/json" }

$body = @{
  topic       = "Email Marketing Best Practices"
  variations  = 3
  tone        = "professional"
  category    = "blog_article"
  language    = "English"
  use_ai      = $true
} | ConvertTo-Json -Compress

Write-Host "POST $BaseUrl/v1/title-meta/generate" -ForegroundColor Yellow
$r = Invoke-RestMethod -Uri "$BaseUrl/v1/title-meta/generate" -Method Post -Headers $Headers -Body $body -TimeoutSec $TimeoutSec

Write-Host ("Topic: {0} | Tone: {1} | Quality: {2}%" -f $r.topic, $r.tone, $r.quality.average_score) -ForegroundColor Green
$r.variations | Select-Object title, title_length, meta_length, quality_score, seo_ready | Format-Table -AutoSize
Write-Host "`n--- Variation 1 ---"
Write-Host "Title: $($r.variations[0].title)"
Write-Host "Meta:  $($r.variations[0].meta_description)"
