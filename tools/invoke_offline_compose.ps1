param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$ComposeArguments
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$composeDirectory = Join-Path $repo "deploy/offline"
$composePath = Join-Path $composeDirectory "compose.yaml"
$envPath = Join-Path $composeDirectory ".env"

function Get-OfflineEnvMap {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Offline environment file is missing: $Path"
    }

    $values = [ordered]@{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.StartsWith("#")) {
            continue
        }
        if ($trimmed -notmatch '^(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?<value>.*)$') {
            throw "Invalid offline environment entry: $line"
        }
        $name = $Matches["name"]
        if ($values.Contains($name)) {
            throw "Offline environment key $name must appear exactly once"
        }
        $values[$name] = $Matches["value"].Trim()
    }
    if ($values.Count -eq 0) {
        throw "Offline environment file is empty: $Path"
    }
    return $values
}

function Assert-SafeComposeArguments {
    param([string[]]$Arguments)

    foreach ($argument in $Arguments) {
        if (
            $argument -in @(
                "-f",
                "--file",
                "--env-file",
                "--project-directory"
            ) -or
            $argument -match '^-f.+' -or
            $argument -match '^--(?:file|env-file|project-directory)='
        ) {
            throw "Compose file, environment, and project-directory override arguments are not allowed"
        }
    }
}

function Normalize-OfflinePath {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "Offline path must not be empty"
    }
    return [IO.Path]::GetFullPath($Path).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
}

function Test-OfflinePathEqual {
    param(
        [string]$Left,
        [string]$Right
    )

    $comparison = [StringComparison]::Ordinal
    if ($env:OS -eq "Windows_NT") {
        $comparison = [StringComparison]::OrdinalIgnoreCase
    }
    return [string]::Equals(
        (Normalize-OfflinePath -Path $Left),
        (Normalize-OfflinePath -Path $Right),
        $comparison
    )
}

function Resolve-OfflineEnvPath {
    param(
        [Collections.IDictionary]$EnvironmentMap,
        [string]$Name
    )

    if (-not $EnvironmentMap.Contains($Name)) {
        throw "$Name must be defined in deploy/offline/.env"
    }
    $rawValue = [string]$EnvironmentMap[$Name]
    if ([string]::IsNullOrWhiteSpace($rawValue)) {
        throw "$Name must not be empty"
    }
    if ($rawValue.Contains("'") -or $rawValue.Contains('"')) {
        throw "$Name must use an unquoted path"
    }
    if ($rawValue -match '^\$\{(?<variable>[A-Za-z_][A-Za-z0-9_]*)\}$') {
        $variableName = $Matches["variable"]
        $rawValue = [Environment]::GetEnvironmentVariable($variableName, "Process")
        if ([string]::IsNullOrWhiteSpace($rawValue)) {
            throw "$Name references missing environment variable $variableName"
        }
    }
    elseif ($rawValue.Contains('$')) {
        throw "$Name uses unsupported Compose variable syntax"
    }

    if ([IO.Path]::IsPathRooted($rawValue)) {
        return Normalize-OfflinePath -Path $rawValue
    }
    return Normalize-OfflinePath -Path (Join-Path $composeDirectory $rawValue)
}

function Get-JsonPropertyValue {
    param(
        [object]$Object,
        [string]$Name
    )

    if ($null -eq $Object) {
        return $null
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
}

function Assert-InternalDigestImage {
    param(
        [string]$Image,
        [string]$Context
    )

    if (
        [string]::IsNullOrWhiteSpace($Image) -or
        $Image -notmatch '^registry\.internal/dc-agent/[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}$'
    ) {
        throw "$Context image must use registry.internal/dc-agent/ and an exact sha256 digest"
    }
}

function Assert-RenderedOfflineCompose {
    param(
        [object]$Rendered,
        [Collections.IDictionary]$EnvironmentMap
    )

    $services = Get-JsonPropertyValue -Object $Rendered -Name "services"
    if ($null -eq $services) {
        throw "Rendered Compose configuration has no services"
    }

    foreach ($serviceProperty in $services.PSObject.Properties) {
        $service = $serviceProperty.Value
        $image = Get-JsonPropertyValue -Object $service -Name "image"
        if ($null -ne $image) {
            Assert-InternalDigestImage -Image ([string]$image) -Context $serviceProperty.Name
        }
        $build = Get-JsonPropertyValue -Object $service -Name "build"
        $args = Get-JsonPropertyValue -Object $build -Name "args"
        $pythonBaseImage = Get-JsonPropertyValue -Object $args -Name "PYTHON_BASE_IMAGE"
        if ($null -ne $build -and $null -eq $pythonBaseImage) {
            throw "$($serviceProperty.Name) build is missing PYTHON_BASE_IMAGE"
        }
        if ($null -ne $pythonBaseImage) {
            Assert-InternalDigestImage -Image ([string]$pythonBaseImage) -Context "$($serviceProperty.Name) PYTHON_BASE_IMAGE"
        }
    }

    $dataRoot = Resolve-OfflineEnvPath -EnvironmentMap $EnvironmentMap -Name "DATA_ROOT"
    $modelRoot = Resolve-OfflineEnvPath -EnvironmentMap $EnvironmentMap -Name "MODEL_ROOT"
    $expectedBinds = [ordered]@{
        "postgres" = [ordered]@{
            "/var/lib/postgresql/data" = Join-Path $dataRoot "postgres"
        }
        "clickhouse" = [ordered]@{
            "/var/lib/clickhouse" = Join-Path $dataRoot "clickhouse"
        }
        "qdrant" = [ordered]@{
            "/qdrant/storage" = Join-Path $dataRoot "qdrant"
        }
        "redis" = [ordered]@{
            "/data" = Join-Path $dataRoot "redis"
        }
        "embedding-service" = [ordered]@{
            "/models" = $modelRoot
        }
        "api" = [ordered]@{
            "/data/raw" = Join-Path $dataRoot "raw"
            "/data/parquet" = Join-Path $dataRoot "parquet"
            "/models" = $modelRoot
        }
        "ingestion-worker" = [ordered]@{
            "/data/raw" = Join-Path $dataRoot "raw"
            "/data/parquet" = Join-Path $dataRoot "parquet"
            "/models" = $modelRoot
        }
        "llama" = [ordered]@{
            "/models" = $modelRoot
        }
    }

    foreach ($serviceProperty in $services.PSObject.Properties) {
        $serviceName = $serviceProperty.Name
        $expectedForService = $null
        if ($expectedBinds.Contains($serviceName)) {
            $expectedForService = $expectedBinds[$serviceName]
        }
        $seenTargets = @{}
        foreach ($volume in @(Get-JsonPropertyValue -Object $serviceProperty.Value -Name "volumes")) {
            if ($null -eq $volume) {
                continue
            }
            if ((Get-JsonPropertyValue -Object $volume -Name "type") -ne "bind") {
                continue
            }
            $target = [string](Get-JsonPropertyValue -Object $volume -Name "target")
            $source = [string](Get-JsonPropertyValue -Object $volume -Name "source")
            if ($null -eq $expectedForService -or -not $expectedForService.Contains($target)) {
                throw "$serviceName has an unexpected bind mount for $target"
            }
            if (-not (Test-OfflinePathEqual -Left $source -Right $expectedForService[$target])) {
                throw "$serviceName bind source for $target does not match deploy/offline/.env"
            }
            $bind = Get-JsonPropertyValue -Object $volume -Name "bind"
            $createHostPath = Get-JsonPropertyValue -Object $bind -Name "create_host_path"
            if ($createHostPath -ne $false) {
                throw "$serviceName bind mount for $target must disable create_host_path"
            }
            $seenTargets[$target] = $true
        }
        if ($null -ne $expectedForService) {
            foreach ($target in $expectedForService.Keys) {
                if (-not $seenTargets.Contains($target)) {
                    throw "$serviceName is missing required bind mount for $target"
                }
            }
        }
    }

    $expectedSecrets = [ordered]@{
        "postgres_password" = [ordered]@{
            EnvName = "POSTGRES_PASSWORD_FILE"
            Path = Join-Path $repo "artifacts/secrets/postgres-password"
        }
        "database_url" = [ordered]@{
            EnvName = "DATABASE_URL_SECRET_FILE"
            Path = Join-Path $repo "artifacts/secrets/database-url"
        }
    }
    $secrets = Get-JsonPropertyValue -Object $Rendered -Name "secrets"
    if ($null -eq $secrets) {
        throw "Rendered Compose configuration has no secrets"
    }
    foreach ($secretName in $expectedSecrets.Keys) {
        $secret = Get-JsonPropertyValue -Object $secrets -Name $secretName
        $renderedPath = [string](Get-JsonPropertyValue -Object $secret -Name "file")
        $configuredPath = Resolve-OfflineEnvPath -EnvironmentMap $EnvironmentMap -Name $expectedSecrets[$secretName]["EnvName"]
        $managedPath = $expectedSecrets[$secretName]["Path"]
        if (
            -not (Test-OfflinePathEqual -Left $configuredPath -Right $managedPath) -or
            -not (Test-OfflinePathEqual -Left $renderedPath -Right $managedPath)
        ) {
            throw "$secretName secret file must use the repository-managed artifacts/secrets path"
        }
    }
}

if ($null -eq $ComposeArguments -or $ComposeArguments.Count -eq 0) {
    throw "Pass Docker Compose arguments, for example: up -d"
}
Assert-SafeComposeArguments -Arguments $ComposeArguments
if (-not (Test-Path -LiteralPath $composePath -PathType Leaf)) {
    throw "Offline Compose file is missing: $composePath"
}

$environmentMap = Get-OfflineEnvMap -Path $envPath
$savedEnvironment = [ordered]@{}
$baseArguments = @(
    "compose",
    "--env-file", $envPath,
    "-f", $composePath
)

try {
    foreach ($name in $environmentMap.Keys) {
        $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
        [Environment]::SetEnvironmentVariable($name, $null, "Process")
    }

    $configErrorPath = [IO.Path]::GetTempFileName()
    $originalConsoleOut = [Console]::Out
    $directConsoleOutput = [IO.StringWriter]::new()
    try {
        [Console]::SetOut($directConsoleOutput)
        $pipelineOutput = & docker @baseArguments "config" "--format" "json" 2> $configErrorPath
        $configExitCode = $LASTEXITCODE
        $configError = [IO.File]::ReadAllText($configErrorPath)
        $directOutputText = $directConsoleOutput.ToString()
    }
    finally {
        [Console]::SetOut($originalConsoleOut)
        $directConsoleOutput.Dispose()
        Remove-Item -LiteralPath $configErrorPath -Force -ErrorAction SilentlyContinue
    }
    if ($configExitCode -ne 0) {
        throw "Docker Compose configuration failed with exit code ${configExitCode}: $configError"
    }
    $configJson = ($pipelineOutput -join [Environment]::NewLine) + $directOutputText
    try {
        $rendered = $configJson | ConvertFrom-Json
    }
    catch {
        throw "Docker Compose configuration did not return valid JSON: $($_.Exception.Message)"
    }
    Assert-RenderedOfflineCompose -Rendered $rendered -EnvironmentMap $environmentMap

    & docker @baseArguments @ComposeArguments
    $composeExitCode = $LASTEXITCODE
    if ($composeExitCode -ne 0) {
        throw "Docker Compose command failed with exit code $composeExitCode"
    }
}
finally {
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], "Process")
    }
}
