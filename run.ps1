param(
    [switch]$SkipInstall,
    [switch]$RebuildIndex,
    [int]$IndexLimit = 0,
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$UseReload
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ImagesDir = Join-Path $ProjectRoot "data\images"
$EmbeddingsPath = Join-Path $ProjectRoot "data\indexes\embeddings.npy"

if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment not found at .venv. Create it first with: py -3.13 -m venv .venv"
}

if (-not $SkipInstall) {
    Write-Host "Installing/updating dependencies..."
    & $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
}

if (-not (Test-Path $ImagesDir)) {
    throw "Image directory not found at data\images. Run .venv\Scripts\python.exe download_images.py first."
}

$ImageCount = (Get-ChildItem $ImagesDir -File -ErrorAction SilentlyContinue | Measure-Object).Count
if ($ImageCount -eq 0) {
    throw "No images found in data\images. Run .venv\Scripts\python.exe download_images.py first."
}

if ($RebuildIndex -or -not (Test-Path $EmbeddingsPath)) {
    Write-Host "Building image index..."
    $buildArgs = @("scripts\build_index.py", "--images-dir", "data\images")
    if ($IndexLimit -gt 0) {
        $buildArgs += @("--limit", $IndexLimit)
    }
    & $VenvPython @buildArgs
}

Write-Host "Starting app at http://$BindHost`:$Port/"
$uvicornArgs = @("-m", "uvicorn", "app.main:app", "--host", $BindHost, "--port", $Port)
if ($UseReload) {
    $uvicornArgs += "--reload"
}
& $VenvPython @uvicornArgs
