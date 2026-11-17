# CityPulse - Launch Cloudflare Quick Tunnel (Public HTTPS URL)
param([int]$Port = 8000)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " CityPulse — Cloudflare Public HTTPS Tunnel" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Check if local server is running
$conn = Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -WarningAction SilentlyContinue
if (-not $conn.TcpTestSucceeded) {
    Write-Host "[!] Warning: CityPulse server does not seem to be running on port $Port." -ForegroundColor Yellow
    Write-Host "    Make sure you have started it in another terminal via: .\run.ps1" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "Launching tunnel for http://localhost:$Port ..." -ForegroundColor Green
Write-Host "Look for the public URL ending with '.trycloudflare.com' below:" -ForegroundColor Green
Write-Host "----------------------------------------------------------" -ForegroundColor DarkGray

.\cloudflared.exe tunnel --url "http://localhost:$Port"
