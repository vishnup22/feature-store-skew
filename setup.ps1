#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvDir = Join-Path $ScriptDir ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$PipExe = Join-Path $VenvDir "Scripts\pip.exe"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-VenvPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    & $PythonExe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: python $($Args -join ' ')"
    }
}

Write-Step "Creating virtual environment with Python 3.10"
try {
    $pyLauncher = Get-Command py -ErrorAction Stop
} catch {
    throw "Python launcher 'py' not found. Install Python 3.10 from https://www.python.org/downloads/"
}

& py -3.10 -m venv $VenvDir
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create virtual environment. Ensure Python 3.10 is installed: py -3.10 --version"
}

Write-Step "Installing Python dependencies"
Invoke-VenvPython -m pip install --upgrade pip
& $PipExe install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install requirements.txt"
}

Write-Step "Downloading NYC taxi parquet dataset"
$downloadScript = @'
from pathlib import Path
import urllib.request

raw_path = Path("data/raw/yellow_tripdata_2023-01.parquet")
url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
raw_path.parent.mkdir(parents=True, exist_ok=True)
print(f"Downloading {url} -> {raw_path}")
urllib.request.urlretrieve(url, raw_path)
print(f"Download complete: {raw_path.stat().st_size} bytes")
'@
Invoke-VenvPython -c $downloadScript

Write-Step "Running PySpark feature engineering"
Invoke-VenvPython pipeline/spark_features.py

Write-Step "Starting Redis for Feast online materialization"
try {
    $docker = Get-Command docker -ErrorAction Stop
    docker compose up -d redis
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose up failed"
    }
    Write-Host "Waiting for Redis to become ready..."
    Start-Sleep -Seconds 3
} catch {
    Write-Warning "Docker not available. Ensure Redis is running on localhost:6379 before materialization."
}

Write-Step "Applying Feast feature definitions (via Docker on Windows)"
try {
    docker compose run --rm --no-deps api bash -c "cd feature_store/feature_repo && feast apply"
    if ($LASTEXITCODE -ne 0) {
        throw "Feast apply failed inside Docker"
    }
} catch {
    throw "Feast apply requires Docker on Windows. Install Docker Desktop and retry."
}

Write-Step "Materializing features to Redis"
Invoke-VenvPython feature_store/materialize.py

Write-Step "Training model and logging feature metadata"
Invoke-VenvPython training/train.py

Write-Host ""
Write-Host "Setup complete. Run 'docker compose up --build' to start the API." -ForegroundColor Green
