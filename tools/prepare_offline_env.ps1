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

function Get-OfflineEnvValue {
    param(
        [string]$Path,
        [string]$Name
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    foreach ($line in Get-Content -LiteralPath $Path) {
        $pattern = "^\s*" + [regex]::Escape($Name) + "\s*=\s*(?<value>.*)\s*$"
        if ($line -match $pattern) {
            $value = $Matches["value"].Trim()
            if ($value.Length -ge 2) {
                $first = $value.Substring(0, 1)
                $last = $value.Substring($value.Length - 1, 1)
                if (($first -eq '"' -and $last -eq '"') -or ($first -eq "'" -and $last -eq "'")) {
                    $value = $value.Substring(1, $value.Length - 2)
                }
            }
            return $value
        }
    }
    return $null
}

function Resolve-OfflineDataRoot {
    $fallback = Join-Path $repo "artifacts/data"
    $rawValue = Get-OfflineEnvValue -Path $envPath -Name "DATA_ROOT"
    if ([string]::IsNullOrWhiteSpace($rawValue) -or $rawValue.Contains('$')) {
        return [IO.Path]::GetFullPath($fallback)
    }
    if ([IO.Path]::IsPathRooted($rawValue)) {
        return [IO.Path]::GetFullPath($rawValue)
    }
    $composeDirectory = Split-Path -Parent $envPath
    return [IO.Path]::GetFullPath((Join-Path $composeDirectory $rawValue))
}

function Invoke-IcaclsChecked {
    param(
        [string]$Path,
        [string[]]$Arguments
    )

    & icacls.exe $Path @Arguments | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to restrict offline secret permissions"
    }
}

function Protect-SecretPath {
    param(
        [string]$Path,
        [switch]$Directory
    )

    if ($env:OS -eq "Windows_NT") {
        $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
        if ($Directory) {
            $acl = New-Object System.Security.AccessControl.DirectorySecurity
            $inheritance = [System.Security.AccessControl.InheritanceFlags]::ContainerInherit -bor [System.Security.AccessControl.InheritanceFlags]::ObjectInherit
        }
        else {
            $acl = New-Object System.Security.AccessControl.FileSecurity
            $inheritance = [System.Security.AccessControl.InheritanceFlags]::None
        }
        $acl.SetAccessRuleProtection($true, $false)
        $rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
            $identity.User,
            [System.Security.AccessControl.FileSystemRights]::FullControl,
            $inheritance,
            [System.Security.AccessControl.PropagationFlags]::None,
            [System.Security.AccessControl.AccessControlType]::Allow
        )
        $acl.AddAccessRule($rule)
        Set-Acl -LiteralPath $Path -AclObject $acl
        Invoke-IcaclsChecked -Path $Path -Arguments @("/inheritance:r")
    }
    else {
        if ($Directory) {
            chmod 700 $Path
        }
        else {
            chmod 600 $Path
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to restrict offline secret permissions"
        }
    }
}

function Remove-SafeSecretLeaf {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $item = Get-Item -LiteralPath $Path -Force
    if ($item.PSIsContainer) {
        throw "Refusing to replace a secret staging directory"
    }
    Remove-Item -LiteralPath $Path -Force
}

function New-OfflinePassword {
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $rng.GetBytes($bytes)
    }
    finally {
        $rng.Dispose()
    }
    return [Convert]::ToBase64String($bytes).Replace('+', '-').Replace('/', '_').TrimEnd('=')
}

function Assert-OfflineSecretPair {
    param(
        [string]$PasswordFile,
        [string]$DatabaseFile
    )

    $password = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $PasswordFile), [Text.Encoding]::ASCII)
    $databaseUrl = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $DatabaseFile), [Text.Encoding]::ASCII)
    if ($password -notmatch '^[A-Za-z0-9_-]{43}$') {
        throw "Generated PostgreSQL password has an invalid format"
    }
    $expectedDatabaseUrl = "postgresql+psycopg://dc_agent:$password@postgres/dc_agent"
    if ($databaseUrl -ne $expectedDatabaseUrl) {
        throw "Generated database URL does not match the PostgreSQL password"
    }
}

function Publish-OfflineSecretPair {
    param(
        [string]$PasswordFile,
        [string]$DatabaseFile,
        [string]$Password,
        [string]$DatabaseUrl
    )

    $stagedPassword = "$PasswordFile.new"
    $stagedDatabase = "$DatabaseFile.new"
    $backupPassword = "$PasswordFile.bak"
    $backupDatabase = "$DatabaseFile.bak"
    $hadPair = (Test-Path -LiteralPath $PasswordFile) -and (Test-Path -LiteralPath $DatabaseFile)
    $passwordBackedUp = $false
    $databaseBackedUp = $false
    $passwordPublished = $false
    $databasePublished = $false
    $published = $false

    try {
        Remove-SafeSecretLeaf -Path $stagedPassword
        Remove-SafeSecretLeaf -Path $stagedDatabase
        if ((Test-Path -LiteralPath $backupPassword) -or (Test-Path -LiteralPath $backupDatabase)) {
            throw "Refusing to rotate while a previous secret backup is present"
        }

        Set-Content -LiteralPath $stagedPassword -Value $Password -NoNewline -Encoding Ascii
        Protect-SecretPath -Path $stagedPassword
        Set-Content -LiteralPath $stagedDatabase -Value $DatabaseUrl -NoNewline -Encoding Ascii
        Protect-SecretPath -Path $stagedDatabase
        Assert-OfflineSecretPair -PasswordFile $stagedPassword -DatabaseFile $stagedDatabase

        if ($hadPair) {
            Move-Item -LiteralPath $PasswordFile -Destination $backupPassword
            $passwordBackedUp = $true
            Move-Item -LiteralPath $DatabaseFile -Destination $backupDatabase
            $databaseBackedUp = $true
        }

        Move-Item -LiteralPath $stagedPassword -Destination $PasswordFile
        $passwordPublished = $true
        Move-Item -LiteralPath $stagedDatabase -Destination $DatabaseFile
        $databasePublished = $true
        Assert-OfflineSecretPair -PasswordFile $PasswordFile -DatabaseFile $DatabaseFile
        Protect-SecretPath -Path $PasswordFile
        Protect-SecretPath -Path $DatabaseFile
        $published = $true
    }
    catch {
        try {
            if ($passwordPublished) {
                Remove-SafeSecretLeaf -Path $PasswordFile
            }
            if ($databasePublished) {
                Remove-SafeSecretLeaf -Path $DatabaseFile
            }
            if ($passwordBackedUp -and -not (Test-Path -LiteralPath $PasswordFile)) {
                Move-Item -LiteralPath $backupPassword -Destination $PasswordFile
            }
            if ($databaseBackedUp -and -not (Test-Path -LiteralPath $DatabaseFile)) {
                Move-Item -LiteralPath $backupDatabase -Destination $DatabaseFile
            }
        }
        catch {
            throw "Secret publication failed and rollback could not be completed"
        }
        throw
    }
    finally {
        Remove-SafeSecretLeaf -Path $stagedPassword
        Remove-SafeSecretLeaf -Path $stagedDatabase
        if ($published) {
            Remove-SafeSecretLeaf -Path $backupPassword
            Remove-SafeSecretLeaf -Path $backupDatabase
        }
    }
}

if (-not (Test-Path -LiteralPath $envPath)) {
    Copy-Item -LiteralPath $envExamplePath -Destination $envPath
}

New-Item -ItemType Directory -Force -Path $secretDir | Out-Null
Protect-SecretPath -Path $secretDir -Directory
$present = @(
    @($passwordPath, $databasePath) |
        Where-Object { Test-Path -LiteralPath $_ }
).Count

if ($present -eq 1) {
    throw "Both offline secret files must exist together; refusing partial configuration"
}

if ($RotateSecrets) {
    $dataRoot = Resolve-OfflineDataRoot
    $pgVersionPath = Join-Path $dataRoot "postgres/PG_VERSION"
    if (Test-Path -LiteralPath $pgVersionPath) {
        throw "Refusing to rotate initialized PostgreSQL secrets. Stop PostgreSQL, perform a controlled ALTER ROLE, update both files, restart, and verify connectivity."
    }
}

if ($present -eq 2) {
    Protect-SecretPath -Path $passwordPath
    Protect-SecretPath -Path $databasePath
    Assert-OfflineSecretPair -PasswordFile $passwordPath -DatabaseFile $databasePath
}

if ($RotateSecrets -or $present -eq 0) {
    $password = New-OfflinePassword
    $databaseUrl = "postgresql+psycopg://dc_agent:$password@postgres/dc_agent"
    Publish-OfflineSecretPair -PasswordFile $passwordPath -DatabaseFile $databasePath -Password $password -DatabaseUrl $databaseUrl
}
