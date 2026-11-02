param(
    [string]$Source = '${NEXUS_RUNTIME_MANIFEST_URL}',
    [string]$StateRoot = "$PSScriptRoot/.nexus-state"
)

$ErrorActionPreference = 'Stop'
$bootstrap = Join-Path $PSScriptRoot 'nexus-bootstrap.exe'
if (!(Test-Path -LiteralPath $bootstrap -PathType Leaf)) { throw "Missing bootstrap executable: $bootstrap" }
if ($env:NEXUS_RUNTIME_MANIFEST_URL) {
    $manifest = [IO.Path]::GetFileName(([Uri]$env:NEXUS_RUNTIME_MANIFEST_URL).AbsolutePath)
    if ($manifest -notmatch '^manifest(?:-[0-9][0-9A-Za-z.+_-]*)?\.json$') { throw 'Invalid NEXUS_RUNTIME_MANIFEST_URL' }
    $StateRoot = Join-Path $StateRoot ([IO.Path]::GetFileNameWithoutExtension($manifest))
}
& $bootstrap --source $Source --state-root $StateRoot --allow-unsigned
exit $LASTEXITCODE
