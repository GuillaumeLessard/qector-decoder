Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Users\Admin\Desktop\qector-build-rc-0.6.2"
Set-Location $RepoRoot

if (-not (Test-Path (Join-Path $RepoRoot "Cargo.toml"))) { throw "Cargo.toml not found" }
if (-not (Test-Path (Join-Path $RepoRoot "pyproject.toml"))) { throw "pyproject.toml not found" }
if (-not (Get-Command maturin -ErrorAction SilentlyContinue)) { throw "maturin not found, pip install maturin" }

& maturin --version

$DistDir = Join-Path $RepoRoot "dist"
if (Test-Path $DistDir) { Remove-Item $DistDir -Recurse -Force }
New-Item -ItemType Directory -Path $DistDir | Out-Null

$pyVers = @("3.9", "3.10", "3.11", "3.12", "3.13")
foreach ($v in $pyVers) {
    Write-Host "Building for Python $v"
    & py -$v -m maturin build --release --strip --features cuda --out $DistDir
}

Get-ChildItem $DistDir/*.whl
Get-FileHash $DistDir/*.whl -Algorithm SHA256 | ForEach-Object {
    "$($_.Hash.ToLower())  $($_.Path | Split-Path -Leaf)"
} | Set-Content $DistDir/SHA256SUMS.txt -Encoding utf8

Get-Content $DistDir/SHA256SUMS.txt
