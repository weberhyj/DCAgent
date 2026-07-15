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
    $values = @()
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
            $values += $value
        }
    }
    if ($values.Count -gt 1) {
        throw "Offline environment key $Name must appear exactly once"
    }
    if ($values.Count -eq 0) {
        return $null
    }
    return $values[0]
}

function Set-OfflineEnvValue {
    param(
        [string]$Path,
        [string]$Name,
        [string]$Value
    )

    $lines = [Collections.Generic.List[string]]::new()
    if (Test-Path -LiteralPath $Path) {
        foreach ($line in Get-Content -LiteralPath $Path) {
            $lines.Add($line)
        }
    }
    $pattern = "^\s*" + [regex]::Escape($Name) + "\s*="
    $matches = @()
    for ($index = 0; $index -lt $lines.Count; $index++) {
        if ($lines[$index] -match $pattern) {
            $matches += $index
        }
    }
    if ($matches.Count -gt 1) {
        throw "Offline environment key $Name must appear exactly once"
    }
    $replacement = "$Name=$Value"
    if ($matches.Count -eq 1) {
        $lines[$matches[0]] = $replacement
    }
    else {
        $lines.Add($replacement)
    }
    [IO.File]::WriteAllLines(
        $Path,
        $lines,
        [Text.UTF8Encoding]::new($false)
    )
}

function Assert-OfflineNumericIdentity {
    param(
        [string]$Name,
        [string]$Value
    )

    if ($Value -notmatch '^[0-9]+$') {
        throw "$Name must contain only decimal digits"
    }
    $number = [Int64]::Parse($Value, [Globalization.CultureInfo]::InvariantCulture)
    if ($number -lt 1 -or $number -gt 2147483647) {
        throw "$Name must be between 1 and 2147483647"
    }
    if ($Value -ne $number.ToString([Globalization.CultureInfo]::InvariantCulture)) {
        throw "$Name must use canonical decimal notation"
    }
    return $Value
}

function Test-OfflineLinuxHost {
    $linuxVariable = Get-Variable -Name IsLinux -ValueOnly -ErrorAction SilentlyContinue
    return $linuxVariable -eq $true
}

function Get-OfflineLinuxIdentity {
    $uid = (& id -u).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to read the current Linux UID"
    }
    $gid = (& id -g).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to read the current Linux GID"
    }
    return [PSCustomObject]@{
        UID = Assert-OfflineNumericIdentity -Name "current Linux UID" -Value $uid
        GID = Assert-OfflineNumericIdentity -Name "current Linux GID" -Value $gid
    }
}

function Assert-OfflineIdentityContract {
    param([switch]$AllowFirstWrite)

    $linuxIdentity = $null
    if (Test-OfflineLinuxHost) {
        $linuxIdentity = Get-OfflineLinuxIdentity
        if ($AllowFirstWrite) {
            Set-OfflineEnvValue -Path $envPath -Name "DCAGENT_UID" -Value $linuxIdentity.UID
            Set-OfflineEnvValue -Path $envPath -Name "DCAGENT_GID" -Value $linuxIdentity.GID
        }
    }

    $uid = Assert-OfflineNumericIdentity -Name "DCAGENT_UID" -Value (
        Get-OfflineEnvValue -Path $envPath -Name "DCAGENT_UID"
    )
    $gid = Assert-OfflineNumericIdentity -Name "DCAGENT_GID" -Value (
        Get-OfflineEnvValue -Path $envPath -Name "DCAGENT_GID"
    )
    foreach ($entry in @(
        @{ Name = "DCAGENT_UID"; Value = $uid },
        @{ Name = "DCAGENT_GID"; Value = $gid }
    )) {
        $override = [Environment]::GetEnvironmentVariable($entry.Name)
        if ($null -ne $override) {
            $override = Assert-OfflineNumericIdentity -Name "$($entry.Name) shell override" -Value $override
            if ($override -ne $entry.Value) {
                throw "$($entry.Name) shell override must match deploy/offline/.env"
            }
        }
    }
    if ($null -ne $linuxIdentity) {
        if ($uid -ne $linuxIdentity.UID -or $gid -ne $linuxIdentity.GID) {
            throw "DCAGENT_UID/DCAGENT_GID must match the current Linux deployment account"
        }
    }
    return [PSCustomObject]@{ UID = $uid; GID = $gid }
}

function Assert-OfflineDeploymentSupport {
    param([object]$Identity)

    if (-not (Test-OfflineLinuxHost)) {
        return
    }
    $dockerHost = [Environment]::GetEnvironmentVariable("DOCKER_HOST")
    if (
        -not [string]::IsNullOrWhiteSpace($dockerHost) -and
        $dockerHost -ne "unix:///var/run/docker.sock"
    ) {
        throw "Only local rootful Linux Docker at /var/run/docker.sock is supported"
    }
    $dockerContext = [Environment]::GetEnvironmentVariable("DOCKER_CONTEXT")
    if (
        -not [string]::IsNullOrWhiteSpace($dockerContext) -and
        $dockerContext -ne "default"
    ) {
        throw "Only the local default Docker context is supported"
    }
    if ($Identity.UID -eq "0" -or $Identity.GID -eq "0") {
        throw "The Linux deployment account must be non-root"
    }
}

function Resolve-OfflineComposePath {
    param([string]$Name)

    $rawValue = Get-OfflineEnvValue -Path $envPath -Name $Name
    if ([string]::IsNullOrWhiteSpace($rawValue)) {
        throw "$Name must be explicitly defined in deploy/offline/.env"
    }
    if ($rawValue -match '^\$\{(?<variable>[A-Za-z_][A-Za-z0-9_]*)\}$') {
        $variableName = $Matches["variable"]
        $expandedValue = [Environment]::GetEnvironmentVariable($variableName)
        if ([string]::IsNullOrWhiteSpace($expandedValue)) {
            throw "$Name references missing environment variable $variableName"
        }
        $rawValue = $expandedValue
    }
    elseif ($rawValue.Contains('$')) {
        throw "$Name uses unsupported or unresolved Compose variable syntax"
    }
    if ($rawValue.Contains('$')) {
        throw "$Name remains unresolved after environment expansion"
    }
    try {
        if ([IO.Path]::IsPathRooted($rawValue)) {
            $resolvedPath = [IO.Path]::GetFullPath($rawValue)
        }
        else {
            $composeDirectory = Split-Path -Parent $envPath
            $resolvedPath = [IO.Path]::GetFullPath((Join-Path $composeDirectory $rawValue))
        }
    }
    catch {
        throw "$Name could not be resolved to a safe host path"
    }

    $shellOverride = [Environment]::GetEnvironmentVariable($Name)
    if ($null -ne $shellOverride) {
        if ([string]::IsNullOrWhiteSpace($shellOverride)) {
            throw "$Name shell override must not be empty"
        }
        if ($shellOverride.Contains('$')) {
            throw "$Name shell override must be one direct host path"
        }
        try {
            if ([IO.Path]::IsPathRooted($shellOverride)) {
                $overridePath = [IO.Path]::GetFullPath($shellOverride)
            }
            else {
                $composeDirectory = Split-Path -Parent $envPath
                $overridePath = [IO.Path]::GetFullPath(
                    (Join-Path $composeDirectory $shellOverride)
                )
            }
        }
        catch {
            throw "$Name shell override could not be resolved to a safe host path"
        }
        $pathsMatch = if ($env:OS -eq "Windows_NT") {
            $overridePath -ieq $resolvedPath
        }
        else {
            $overridePath -ceq $resolvedPath
        }
        if (-not $pathsMatch) {
            throw "$Name shell override must resolve to the same path as deploy/offline/.env"
        }
    }
    return $resolvedPath
}

function Resolve-OfflineDataRoot {
    return Resolve-OfflineComposePath -Name "DATA_ROOT"
}

function Resolve-OfflineModelRoot {
    return Resolve-OfflineComposePath -Name "MODEL_ROOT"
}

function Assert-OfflineExpectedPath {
    param(
        [string]$Name,
        [string]$ActualPath,
        [string]$ExpectedPath
    )

    $expectedFullPath = [IO.Path]::GetFullPath($ExpectedPath)
    $pathsMatch = if ($env:OS -eq "Windows_NT") {
        $ActualPath -ieq $expectedFullPath
    }
    else {
        $ActualPath -ceq $expectedFullPath
    }
    if (-not $pathsMatch) {
        throw "$Name must resolve to the repository-managed secret path"
    }
}

function Assert-OfflinePathIsNotLink {
    param([string]$Path)

    $item = Get-Item -LiteralPath $Path -Force
    if ($null -ne $item.LinkType -or $item.Attributes.HasFlag([IO.FileAttributes]::ReparsePoint)) {
        throw "Offline paths must not be symbolic links or reparse points: $Path"
    }
}

function Get-OfflineLinuxStat {
    param(
        [string]$Path,
        [string]$Format
    )

    $value = (& stat -c $Format -- $Path).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to inspect Linux path metadata: $Path"
    }
    return $value
}

function Assert-OfflineLinuxPathContract {
    param(
        [string]$Path,
        [string]$ExpectedMode,
        [string]$ExpectedUID,
        [string]$ExpectedGID
    )

    Assert-OfflinePathIsNotLink -Path $Path
    $actualUID = Get-OfflineLinuxStat -Path $Path -Format "%u"
    $actualGID = Get-OfflineLinuxStat -Path $Path -Format "%g"
    $actualMode = Get-OfflineLinuxStat -Path $Path -Format "%a"
    if (
        $actualUID -ne $ExpectedUID -or
        $actualGID -ne $ExpectedGID -or
        $actualMode -ne $ExpectedMode
    ) {
        throw "Offline path owner or mode is unsafe: $Path"
    }
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

    Assert-OfflinePathIsNotLink -Path $Path
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

function Initialize-OfflineBindDirectories {
    param(
        [string]$DataRoot,
        [string]$ModelRoot,
        [object]$Identity
    )

    $vendorDirectories = @(
        (Join-Path $DataRoot "postgres"),
        (Join-Path $DataRoot "clickhouse"),
        (Join-Path $DataRoot "qdrant"),
        (Join-Path $DataRoot "redis")
    )
    if (Test-Path -LiteralPath $DataRoot) {
        Assert-OfflinePathIsNotLink -Path $DataRoot
    }
    foreach ($directory in $vendorDirectories) {
        if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
            throw "Vendor bind source must be pre-created with locked-image ownership: $directory"
        }
        Assert-OfflinePathIsNotLink -Path $directory
    }
    if (-not (Test-Path -LiteralPath $ModelRoot -PathType Container)) {
        throw "Model bind source must be pre-created: $ModelRoot"
    }
    Assert-OfflinePathIsNotLink -Path $ModelRoot

    $rawRoot = Join-Path $DataRoot "raw"
    $parquetRoot = Join-Path $DataRoot "parquet"
    $missingDirectories = @()
    foreach ($directory in @($rawRoot, $parquetRoot)) {
        if (Test-Path -LiteralPath $directory) {
            if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
                throw "Writable bind source must be a directory: $directory"
            }
            Assert-OfflinePathIsNotLink -Path $directory
            if (Test-OfflineLinuxHost) {
                Assert-OfflineLinuxPathContract -Path $directory -ExpectedMode "700" -ExpectedUID $Identity.UID -ExpectedGID $Identity.GID
            }
        }
        else {
            $missingDirectories += $directory
        }
    }

    $createdDirectories = @()
    try {
        foreach ($directory in $missingDirectories) {
            New-Item -ItemType Directory -Path $directory | Out-Null
            $createdDirectories += $directory
        }
        foreach ($directory in $createdDirectories) {
            Protect-SecretPath -Path $directory -Directory
        }
    }
    catch {
        foreach ($directory in $createdDirectories) {
            if (Test-Path -LiteralPath $directory -PathType Container) {
                Remove-Item -LiteralPath $directory -Force
            }
        }
        throw
    }
    if (Test-OfflineLinuxHost) {
        Assert-OfflineLinuxPathContract -Path $rawRoot -ExpectedMode "700" -ExpectedUID $Identity.UID -ExpectedGID $Identity.GID
        Assert-OfflineLinuxPathContract -Path $parquetRoot -ExpectedMode "700" -ExpectedUID $Identity.UID -ExpectedGID $Identity.GID
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

    Assert-OfflinePathIsNotLink -Path $PasswordFile
    Assert-OfflinePathIsNotLink -Path $DatabaseFile
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

$envWasCreated = $false
if (-not (Test-Path -LiteralPath $envPath)) {
    Copy-Item -LiteralPath $envExamplePath -Destination $envPath
    $envWasCreated = $true
}

$offlineIdentity = Assert-OfflineIdentityContract -AllowFirstWrite:$envWasCreated
Assert-OfflineDeploymentSupport -Identity $offlineIdentity
$dataRoot = Resolve-OfflineDataRoot
$modelRoot = Resolve-OfflineModelRoot
$configuredPasswordPath = Resolve-OfflineComposePath -Name "POSTGRES_PASSWORD_FILE"
$configuredDatabasePath = Resolve-OfflineComposePath -Name "DATABASE_URL_SECRET_FILE"
Assert-OfflineExpectedPath -Name "POSTGRES_PASSWORD_FILE" -ActualPath $configuredPasswordPath -ExpectedPath $passwordPath
Assert-OfflineExpectedPath -Name "DATABASE_URL_SECRET_FILE" -ActualPath $configuredDatabasePath -ExpectedPath $databasePath
if (Test-Path -LiteralPath $dataRoot) {
    Assert-OfflinePathIsNotLink -Path $dataRoot
}
$present = @(
    @($passwordPath, $databasePath) |
        Where-Object { Test-Path -LiteralPath $_ }
).Count

if ($present -eq 1) {
    throw "Both offline secret files must exist together; refusing partial configuration"
}

if (Test-Path -LiteralPath $secretDir) {
    Assert-OfflinePathIsNotLink -Path $secretDir
}
if ($present -eq 2) {
    Assert-OfflinePathIsNotLink -Path $passwordPath
    Assert-OfflinePathIsNotLink -Path $databasePath
}
if (Test-OfflineLinuxHost) {
    if (Test-Path -LiteralPath $secretDir) {
        Assert-OfflineLinuxPathContract -Path $secretDir -ExpectedMode "700" -ExpectedUID $offlineIdentity.UID -ExpectedGID $offlineIdentity.GID
    }
    if ($present -eq 2) {
        Assert-OfflineLinuxPathContract -Path $passwordPath -ExpectedMode "600" -ExpectedUID $offlineIdentity.UID -ExpectedGID $offlineIdentity.GID
        Assert-OfflineLinuxPathContract -Path $databasePath -ExpectedMode "600" -ExpectedUID $offlineIdentity.UID -ExpectedGID $offlineIdentity.GID
    }
}

if ($RotateSecrets) {
    $pgVersionPath = Join-Path $dataRoot "postgres/PG_VERSION"
    if (Test-Path -LiteralPath $pgVersionPath) {
        throw "Refusing to rotate initialized PostgreSQL secrets. Stop PostgreSQL, perform a controlled ALTER ROLE, update both files, restart, and verify connectivity."
    }
}

Initialize-OfflineBindDirectories -DataRoot $dataRoot -ModelRoot $modelRoot -Identity $offlineIdentity
New-Item -ItemType Directory -Force -Path $secretDir | Out-Null
Protect-SecretPath -Path $secretDir -Directory
if (Test-OfflineLinuxHost) {
    Assert-OfflineLinuxPathContract -Path $secretDir -ExpectedMode "700" -ExpectedUID $offlineIdentity.UID -ExpectedGID $offlineIdentity.GID
}

if ($present -eq 2) {
    Protect-SecretPath -Path $passwordPath
    Protect-SecretPath -Path $databasePath
    Assert-OfflineSecretPair -PasswordFile $passwordPath -DatabaseFile $databasePath
    if (Test-OfflineLinuxHost) {
        Assert-OfflineLinuxPathContract -Path $passwordPath -ExpectedMode "600" -ExpectedUID $offlineIdentity.UID -ExpectedGID $offlineIdentity.GID
        Assert-OfflineLinuxPathContract -Path $databasePath -ExpectedMode "600" -ExpectedUID $offlineIdentity.UID -ExpectedGID $offlineIdentity.GID
    }
}

if ($RotateSecrets -or $present -eq 0) {
    $password = New-OfflinePassword
    $databaseUrl = "postgresql+psycopg://dc_agent:$password@postgres/dc_agent"
    Publish-OfflineSecretPair -PasswordFile $passwordPath -DatabaseFile $databasePath -Password $password -DatabaseUrl $databaseUrl
    if (Test-OfflineLinuxHost) {
        Assert-OfflineLinuxPathContract -Path $passwordPath -ExpectedMode "600" -ExpectedUID $offlineIdentity.UID -ExpectedGID $offlineIdentity.GID
        Assert-OfflineLinuxPathContract -Path $databasePath -ExpectedMode "600" -ExpectedUID $offlineIdentity.UID -ExpectedGID $offlineIdentity.GID
    }
}
