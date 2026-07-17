$ErrorActionPreference = "Stop"
$ComposeArguments = @($args)

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

    $command = $null
    $commandIndex = -1
    for ($index = 0; $index -lt $Arguments.Count; $index++) {
        $argument = $Arguments[$index]
        if ($null -ne $command) { break }
        if (
            $argument -in @(
                "-f",
                "--file",
                "--env-file",
                "--project-directory",
                "-p",
                "--project-name"
            ) -or
            $argument -match '^-f.+' -or
            $argument -match '^-p.+' -or
            $argument -match '^--(?:file|env-file|project-directory|project-name)='
        ) {
            throw "Compose file, environment, project-directory, and project-name override arguments are not allowed"
        }
        if ($argument -in @("--ansi", "--parallel", "--profile", "--progress")) {
            if ($index + 1 -ge $Arguments.Count) {
                throw "Compose global option $argument requires a value"
            }
            $index++
            continue
        }
        if ($argument -match '^--(?:ansi|parallel|profile|progress)=') {
            continue
        }
        if ($argument -in @("--compatibility", "--dry-run")) {
            continue
        }
        if ($argument.StartsWith("-")) {
            throw "Unsupported Compose global option $argument could bypass preflight validation"
        }
        $command = $argument
        $commandIndex = $index
    }
    if ($null -eq $command) {
        throw "A Docker Compose command is required"
    }
    if ($command -in @("create", "restart", "run", "scale", "start")) {
        throw "docker compose $command is not allowed because lifecycle overrides bypass the validated service model"
    }
    if ($command -eq "build") {
        for ($index = $commandIndex + 1; $index -lt $Arguments.Count; $index++) {
            $argument = $Arguments[$index]
            if (
                $argument -in @(
                    "--build-arg",
                    "--build-context",
                    "--builder",
                    "--secret",
                    "--ssh"
                ) -or
                $argument -match '^--(?:build-arg|build-context|builder|secret|ssh)='
            ) {
                throw "Compose build override argument $argument is not allowed"
            }
        }
    }
    if ($command -eq "up") {
        for ($index = $commandIndex + 1; $index -lt $Arguments.Count; $index++) {
            $argument = $Arguments[$index]
            if ($argument -in @("--no-build", "--no-deps", "--no-recreate", "--scale") -or $argument -match '^--scale=') {
                throw "Compose lifecycle override argument $argument is not allowed"
            }
        }
    }
}

function Assert-LocalDockerContext {
    $dockerHost = [Environment]::GetEnvironmentVariable("DOCKER_HOST", "Process")
    if (
        -not [string]::IsNullOrWhiteSpace($dockerHost) -and
        $dockerHost -cne "unix:///var/run/docker.sock"
    ) {
        throw "Only the local rootful Docker host at unix:///var/run/docker.sock is supported"
    }
    $dockerContext = [Environment]::GetEnvironmentVariable(
        "DOCKER_CONTEXT",
        "Process"
    )
    if (
        -not [string]::IsNullOrWhiteSpace($dockerContext) -and
        $dockerContext -cne "default"
    ) {
        throw "Only the local default Docker context is supported"
    }
}

function Assert-LocalDockerEndpoint {
    $contextErrorPath = [IO.Path]::GetTempFileName()
    $savedErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $contextOutput = & docker "context" "inspect" "default" "--format" "{{.Endpoints.docker.Host}}" 2> $contextErrorPath
        $contextExitCode = $LASTEXITCODE
        $contextError = [IO.File]::ReadAllText($contextErrorPath)
    }
    finally {
        $ErrorActionPreference = $savedErrorActionPreference
        Remove-Item -LiteralPath $contextErrorPath -Force -ErrorAction SilentlyContinue
    }
    if ($contextExitCode -ne 0) {
        throw "Docker default context could not be inspected: $contextError"
    }
    $endpoint = ($contextOutput -join [Environment]::NewLine).Trim()
    if ($endpoint -cne "unix:///var/run/docker.sock") {
        throw "Docker default context must use unix:///var/run/docker.sock"
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
        $Image -cnotmatch '^registry\.internal/dc-agent/[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}$'
    ) {
        throw "$Context image must use registry.internal/dc-agent/ and an exact sha256 digest"
    }
}

function Assert-RenderedOfflineCompose {
    param(
        [object]$Rendered,
        [Collections.IDictionary]$EnvironmentMap
    )

    $projectName = [string](Get-JsonPropertyValue -Object $Rendered -Name "name")
    if ($projectName -cne "dc-agent-offline") {
        throw "Rendered Compose project name must be dc-agent-offline"
    }
    $services = Get-JsonPropertyValue -Object $Rendered -Name "services"
    if ($null -eq $services) {
        throw "Rendered Compose configuration has no services"
    }
    foreach ($requiredService in @(
        "postgres",
        "clickhouse",
        "qdrant",
        "redis",
        "clamav",
        "schema-migration",
        "embedding-service",
        "api",
        "ingestion-worker",
        "llama"
    )) {
        if ($null -eq (Get-JsonPropertyValue -Object $services -Name $requiredService)) {
            throw "Rendered Compose configuration is missing required service $requiredService"
        }
    }

    foreach ($serviceProperty in $services.PSObject.Properties) {
        $service = $serviceProperty.Value
        $rawPorts = Get-JsonPropertyValue -Object $service -Name "ports"
        $ports = @()
        if ($null -ne $rawPorts) {
            $ports = @($rawPorts)
        }
        if ($serviceProperty.Name -eq "api") {
            if ($ports.Count -ne 1) {
                throw "api must publish exactly one loopback port"
            }
            $port = $ports[0]
            $hostIp = [string](Get-JsonPropertyValue -Object $port -Name "host_ip")
            $published = [string](Get-JsonPropertyValue -Object $port -Name "published")
            $target = [string](Get-JsonPropertyValue -Object $port -Name "target")
            $protocol = [string](Get-JsonPropertyValue -Object $port -Name "protocol")
            if (
                $hostIp -cne "127.0.0.1" -or
                $published -cne "8000" -or
                $target -cne "8000" -or
                $protocol -cne "tcp"
            ) {
                throw "api port must be 127.0.0.1:8000:8000/tcp"
            }
        }
        elseif ($ports.Count -ne 0) {
            throw "$($serviceProperty.Name) must not publish ports"
        }
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
Assert-LocalDockerContext
if (-not (Test-Path -LiteralPath $composePath -PathType Leaf)) {
    throw "Offline Compose file is missing: $composePath"
}

$environmentMap = Get-OfflineEnvMap -Path $envPath
$savedEnvironment = [ordered]@{}
$cleanEnvironmentNames = @(
    @($environmentMap.Keys)
    "COMPOSE_ENV_FILES"
    "COMPOSE_FILE"
    "COMPOSE_PROFILES"
    "COMPOSE_PROJECT_NAME"
) | Sort-Object -Unique
$baseArguments = @(
    "--context", "default",
    "compose",
    "--env-file", $envPath,
    "-f", $composePath
)

try {
    foreach ($name in $cleanEnvironmentNames) {
        $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
        [Environment]::SetEnvironmentVariable($name, $null, "Process")
    }

    Assert-LocalDockerEndpoint

    $configErrorPath = [IO.Path]::GetTempFileName()
    $originalConsoleOut = [Console]::Out
    $directConsoleOutput = [IO.StringWriter]::new()
    $configErrorActionPreference = $ErrorActionPreference
    try {
        [Console]::SetOut($directConsoleOutput)
        $ErrorActionPreference = "Continue"
        $pipelineOutput = & docker @baseArguments "--profile" "*" "config" "--format" "json" 2> $configErrorPath
        $configExitCode = $LASTEXITCODE
        $configError = [IO.File]::ReadAllText($configErrorPath)
        $directOutputText = $directConsoleOutput.ToString()
    }
    finally {
        $ErrorActionPreference = $configErrorActionPreference
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
