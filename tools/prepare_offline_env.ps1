param(
    [switch]$RotateSecrets
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$envExamplePath = Join-Path $repo "deploy/offline/.env.example"
$envPath = Join-Path $repo "deploy/offline/.env"
$secretDir = Join-Path $repo "artifacts/secrets"
$passwordPath = Join-Path $secretDir "postgres-password"
$databasePath = Join-Path $secretDir "database-url"

if (-not (Test-Path -LiteralPath $envPath)) {
    Copy-Item -LiteralPath $envExamplePath -Destination $envPath
}

New-Item -ItemType Directory -Force -Path $secretDir | Out-Null
$present = @(
    @($passwordPath, $databasePath) |
        Where-Object { Test-Path -LiteralPath $_ }
).Count

if ($present -eq 1) {
    throw "Both offline secret files must exist together; refusing partial configuration"
}

if ($RotateSecrets -or $present -eq 0) {
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $rng.GetBytes($bytes)
    }
    finally {
        $rng.Dispose()
    }

    $password = [Convert]::ToBase64String($bytes).Replace('+', '-').Replace('/', '_').TrimEnd('=')
    Set-Content -LiteralPath $passwordPath -Value $password -NoNewline -Encoding Ascii
    Set-Content -LiteralPath $databasePath -Value "postgresql+psycopg://dc_agent:$password@postgres/dc_agent" -NoNewline -Encoding Ascii
}

if ($env:OS -eq "Windows_NT") {
    $account = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    & icacls.exe $secretDir "/inheritance:r" "/grant:r" "${account}:(OI)(CI)F" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to restrict offline secret directory permissions"
    }
    foreach ($secretPath in @($passwordPath, $databasePath)) {
        & icacls.exe $secretPath "/inheritance:r" "/grant:r" "${account}:F" | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to restrict offline secret file permissions"
        }
    }
}
else {
    chmod 700 $secretDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to restrict offline secret directory permissions"
    }
    chmod 600 $passwordPath $databasePath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to restrict offline secret file permissions"
    }
}
