param(
    [string]$Python = "python",
    [switch]$Inspect
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if ($Inspect) {
    & $Python export_tool.py --inspect
} else {
    & $Python export_tool.py
}
