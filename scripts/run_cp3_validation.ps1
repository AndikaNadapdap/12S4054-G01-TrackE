# CP3: run every validation configuration on ALL evaluable users (subset: 0).
# Run from the repository root, from COMMITTED code (check `git status` first).
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_cp3_validation.ps1 -DataDir "D:\sisrek_data"
param([Parameter(Mandatory=$true)][string]$DataDir)
$ErrorActionPreference = "Stop"
New-Item -ItemType Junction -Path "MAIN DATA" -Target (Join-Path $DataDir "MAIN DATA") -Force | Out-Null
New-Item -ItemType Junction -Path "SUPPORTING_DATA" -Target (Join-Path $DataDir "SUPPORTING_DATA") -Force | Out-Null
$configs = @("configs\e1_popularity.yaml") + (Get-ChildItem configs\e2_grid\*.yaml | ForEach-Object { $_.FullName }) +
           (Get-ChildItem configs\e3[abc]_*.yaml, configs\e4[abc]_*.yaml | ForEach-Object { $_.FullName })
foreach ($cfg in $configs) { python -m src.run --config $cfg; if ($LASTEXITCODE -ne 0) { throw "failed: $cfg" } }
Remove-Item "MAIN DATA", "SUPPORTING_DATA" -Force
python scripts\select_best.py
