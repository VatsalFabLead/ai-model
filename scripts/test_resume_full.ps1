# Full Resume - single API: POST /v1/resume-builder/generate
# Usage: .\scripts\test_resume_full.ps1

param(
  [string]$BaseUrl = "http://127.0.0.1:8000",
  [string]$ApiKey = "wYGt5Pq-fUUVzO9mWVtkhoVwLWBWPjEY8InlnWqoqDshkVl2",
  [int]$TimeoutSec = 180
)

$ErrorActionPreference = "Stop"
$Headers = @{
  "Authorization" = "Bearer $ApiKey"
  "Content-Type"  = "application/json"
}

Write-Host "POST $BaseUrl/v1/resume-builder/generate" -ForegroundColor Yellow
Write-Host "Single API - all fields + AI in one call`n" -ForegroundColor Cyan

$body = @{
  full_name      = "Vatsal Patel"
  job_title      = "Flutter Developer"
  email          = "vatsal.fablead@gmail.com"
  phone          = "+91-9876543210"
  linkedin       = "https://linkedin.com/in/vatsalpatel"
  portfolio      = "https://github.com/VatsalFabLead"
  education      = "### B.Tech CS - GTU`n*2018-2022*`n- CGPA: 8.6"
  experience     = "### Flutter Developer - Fablead`n- Built Flutter apps`n- Integrated Firebase"
  skills         = $null
  summary        = $null
  projects       = "- E-Commerce App - Flutter + Firebase`n- Health Tracker - 10K+ downloads"
  certifications = "- Google Flutter (2023)`n- AWS Cloud Practitioner (2024)"
  achievements   = "- Hackathon Winner 2021"
  languages      = "- English - Fluent`n- Hindi - Native"
  template       = "modern"
  language       = "English"
  use_ai         = $true
} | ConvertTo-Json -Depth 4 -Compress

$r = Invoke-RestMethod -Uri "$BaseUrl/v1/resume-builder/generate" -Method Post -Headers $Headers -Body $body -TimeoutSec $TimeoutSec

Write-Host "--- Response ---" -ForegroundColor Green
Write-Host ("Name:     {0} - {1}" -f $r.full_name, $r.job_title)
Write-Host ("Category: {0} | Template: {1} | Lang: {2}" -f $r.category, $r.template, $r.language)
Write-Host ("Quality:  {0}% | Ready: {1}" -f $r.quality.completeness_score, $r.quality.resume_ready)
Write-Host ("AI: skills={0} summary={1} experience={2}" -f $r.ai.skills_generated, $r.ai.summary_generated, $r.ai.experience_enhanced)
Write-Host ("Skills:   {0}" -f ($r.skills_list -join ', '))
Write-Host "`nSummary:`n$($r.summary)"
Write-Host "`n--- Resume Markdown ---`n$($r.resume_markdown)"

$outFile = Join-Path (Split-Path $PSScriptRoot -Parent) "resume_output.md"
$r.resume_markdown | Out-File -FilePath $outFile -Encoding utf8
Write-Host "`nSaved: $outFile" -ForegroundColor Green
