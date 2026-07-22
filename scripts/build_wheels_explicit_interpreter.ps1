Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Users\Admin\Desktop\qector-build-rc-0.6.2"
Set-Location $RepoRoot

if (-not (Get-Command maturin -ErrorAction SilentlyContinue)) { throw "maturin not found" }
& maturin --version

$DistDir = Join-Path $RepoRoot "dist"
if (Test-Path $DistDir) { Remove-Item $DistDir -Recurse -Force }
New-Item -ItemType Directory -Path $DistDir | Out-Null

# Explicit interpreter paths -- 'py -X.Y -m maturin' does NOT reliably force
# maturin's own interpreter discovery; maturin does its own PATH resolution
# and silently rebuilds for whatever python.exe it finds first (observed:
# always resolved to 3.12 regardless of the py launcher version requested).
# Passing --interpreter with the full path is the only reliable fix.
$interpreters = @{
    "3.9"  = "C:\Users\Admin\AppData\Local\Programs\Python\Python39\python.exe"
    "3.10" = "C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe"
    "3.11" = "C:\Users\Admin\AppData\Local\Programs\Python\Python311\python.exe"
    "3.12" = "C:\Program Files\Python312\python.exe"
    "3.13" = "C:\Program Files\Python313\python.exe"
}

foreach ($v in $interpreters.Keys) {
    $exe = $interpreters[$v]
    if (-not (Test-Path $exe)) { throw "Interpreter not found for $v : $exe" }
    Write-Host "=== Building for Python $v ($exe) ==="
    & maturin build --release --strip --features cuda --interpreter $exe --out $DistDir
    if ($LASTEXITCODE -ne 0) { throw "maturin build failed for Python $v (exit $LASTEXITCODE)" }
}

Write-Host "=== Wheels built ==="
Get-ChildItem $DistDir/*.whl | ForEach-Object { $_.Name }
