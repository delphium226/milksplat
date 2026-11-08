# Builds the portable release zip: dist/milksplat-<version>-win64.zip
# Works under Windows PowerShell 5.1 and pwsh.
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    $env:PATH = (Join-Path $env:USERPROFILE '.cargo\bin') + [IO.Path]::PathSeparator + $env:PATH
}
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw 'cargo was not found on PATH or in %USERPROFILE%\.cargo\bin'
}

# Version of the milksplat crate (it inherits the workspace version).
$meta = (cargo metadata --format-version 1 --no-deps) | Out-String | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'cargo metadata failed' }
$version = ($meta.packages | Where-Object { $_.name -eq 'milksplat' }).version
if (-not $version) { throw 'could not find the milksplat package version' }
$name = "milksplat-$version-win64"
Write-Host "Packaging $name"

Write-Host '== cargo test --workspace'
cargo test --workspace
if ($LASTEXITCODE -ne 0) { throw 'cargo test failed' }

Write-Host '== cargo build --release -p milksplat'
cargo build --release -p milksplat
if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' }

$dist = Join-Path $root 'dist'
$stage = Join-Path $dist $name
$zip = Join-Path $dist "$name.zip"
New-Item -ItemType Directory -Force -Path $dist | Out-Null
Get-ChildItem -Path $dist -Filter "$name*" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force

New-Item -ItemType Directory -Force -Path $stage | Out-Null
$exe = Join-Path $root 'target\release\milksplat.exe'
Copy-Item $exe $stage
New-Item -ItemType Directory -Force -Path (Join-Path $stage 'presets') | Out-Null
Get-ChildItem (Join-Path $root 'presets') -File |
    Where-Object { $_.Name -ne 'default.splat' } |
    Copy-Item -Destination (Join-Path $stage 'presets')
Copy-Item (Join-Path $root 'textures') -Destination $stage -Recurse
foreach ($f in 'LICENSE-MIT', 'LICENSE-APACHE', 'README.md') {
    Copy-Item (Join-Path $root $f) $stage
}

Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal
Write-Host ("Created {0} ({1:N1} MB)" -f $zip, ((Get-Item $zip).Length / 1MB))

# Static CRT check: the exe must not depend on the VC++ runtime DLLs.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$dumpbin = $null
if (Test-Path $vswhere) {
    $dumpbin = & $vswhere -latest -products * -find 'VC\Tools\MSVC\**\bin\Hostx64\x64\dumpbin.exe' |
        Select-Object -First 1
}
if (-not $dumpbin) {
    Write-Warning 'dumpbin.exe not found; skipping the static CRT check'
} else {
    Write-Host "== dumpbin /dependents ($dumpbin)"
    $out = & $dumpbin /dependents (Join-Path $stage 'milksplat.exe')
    if ($LASTEXITCODE -ne 0) { throw 'dumpbin failed' }
    $out | ForEach-Object { Write-Host $_ }
    $bad = $out | Select-String -Pattern 'VCRUNTIME140|MSVCP140'
    if ($bad) { throw "the exe depends on the VC++ runtime: $($bad -join '; ')" }
    Write-Host 'Static CRT check passed.'
}
Write-Host "Done: $zip"
