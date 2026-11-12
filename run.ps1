# CityPulse - start the full-stack app (Windows PowerShell)
# Usage:  .\run.ps1            (defaults to port 8000)
#         .\run.ps1 -Port 8001
param([int]$Port = 8000)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not $env:CONDA_DEFAULT_ENV -or $env:CONDA_DEFAULT_ENV -ne "citypulse") {
    Write-Host "Tip: activate the env first ->  conda activate citypulse" -ForegroundColor Yellow
}

Write-Host "Starting CityPulse on http://localhost:$Port  (docs at /docs)" -ForegroundColor Cyan
uvicorn backend.main:app --reload --port $Port
