# Supervisor Structured Worker Manager Design

## Context

The non-Docker Ubuntu deployment runs DC-Agent processes under Supervisor. Structured XLSX
publication depends on the long-running `app.structured_worker` process, but the repository does
not currently provide a Supervisor configuration or an installation command for it. Starting the
module manually leaves queued jobs unprocessed after the terminal closes or the process exits.

## Goal

Provide one repository-managed Bash script that installs an idempotent Supervisor program for the
structured worker and exposes a small lifecycle interface for operators.

The operator-facing entry point will be:

```bash
sudo ./tools/manage_structured_worker_supervisor.sh install \
  --project-root /opt/DCAgent \
  --user dcagent \
  --env-file /etc/dcagent/dcagent.env
```

After installation, the same script will support `start`, `stop`, `restart`, `status`, and `logs`.

## Non-goals

- Do not install the Supervisor OS package.
- Do not manage the API, retrieval worker, embedding service, reranker, or databases.
- Do not create, copy, edit, or print application secrets.
- Do not add an uninstall command or delete an existing Supervisor configuration.
- Do not change the structured worker implementation, queue semantics, or database schema.

## Script interface

The new file will be `tools/manage_structured_worker_supervisor.sh`.

### Install command

```text
install --project-root PATH --user USER --env-file PATH [--python PATH]
```

- `--project-root` must contain `backend/app/structured_worker.py`.
- `--user` is the non-root account used by Supervisor to run the worker.
- `--env-file` must be a readable, shell-compatible environment file shared with the API process.
- `--python` defaults to `<project-root>/backend/.venv/bin/python`.

Install requires root because it writes `/etc/supervisor/conf.d/dcagent-structured-worker.conf`
and creates `/var/log/dcagent`. It validates all inputs before modifying Supervisor state.

### Lifecycle commands

```text
start
stop
restart
status
logs
```

These commands operate on the fixed Supervisor program name `dcagent-structured-worker`. The
`logs` command follows both the standard output and error logs until interrupted.

### Internal run command

The generated Supervisor configuration invokes the same script through a non-public `run`
subcommand. `run` receives the validated project root, environment file, and Python path, exports
the environment file values without printing them, changes to the backend directory, and ends with:

```bash
exec "$python_path" -m app.structured_worker
```

Using `exec` gives Supervisor direct ownership of the Python worker process and preserves signal
delivery and exit status.

### Configuration rendering command

```text
render-config --project-root PATH --user USER --env-file PATH [--python PATH]
```

This diagnostic command performs input validation and writes the proposed Supervisor configuration
to standard output without requiring root, writing system files, or calling `supervisorctl`. It is
used by automated tests and lets an operator inspect the generated configuration before installing
it. It prints the environment file path but never reads or prints the file contents.

## Generated Supervisor configuration

The installed program will have these properties:

- `directory=<project-root>/backend`
- `command=<manager-script> run ...`
- `user=<configured-user>`
- `autostart=true`
- `autorestart=true`
- bounded startup retries and a non-zero `startsecs`
- `stopsignal=TERM`, `stopasgroup=true`, and `killasgroup=true`
- unbuffered Python output
- separate rotating logs under `/var/log/dcagent`

The configuration will not contain secret values. It references the environment file by absolute
path. The environment file must already be readable by the configured service user.

## Installation and update flow

1. Require Linux, Bash, root privileges, `supervisorctl`, and a Supervisor configuration directory.
2. Resolve all supplied paths to absolute paths and reject newlines or unsupported values.
3. Verify the worker module, Python executable, manager script, service user, and environment file.
4. Verify the service user can read the project, Python executable, manager script, and environment
   file.
5. Create `/var/log/dcagent` with ownership assigned to the service user.
6. Render the Supervisor configuration to a temporary file.
7. Compare it with the installed configuration and replace the destination atomically only when it
   differs.
8. Run `supervisorctl reread` and `supervisorctl update`.
9. Start the program if stopped, or restart it when installation changed an existing configuration.
10. Print `supervisorctl status dcagent-structured-worker` without printing application variables.

Repeating the same install command is safe. An unchanged configuration does not trigger an
unnecessary restart.

## Error handling

The script uses strict Bash mode and exits non-zero with a concise error when validation or a
Supervisor operation fails. It must not report a successful install until Supervisor reports the
program as running. Existing configuration remains intact if rendering or validation fails.

The script does not silently fall back to another Python interpreter or environment file. This is
important because an API and worker using different database settings can leave jobs permanently in
the `queued` state.

## Security

- Never use `eval` to process arguments or environment values.
- Never echo, trace, or copy the environment file contents.
- Require the environment file to be a regular file that is not group- or world-writable.
- Quote every operator-provided path and validate the service user separately.
- Write the Supervisor configuration as root and keep secret material outside the repository.
- Run the worker as the configured non-root account.
- Avoid accepting arbitrary Supervisor program names or configuration destinations.

## Testing

Add a Python contract test under `tools/tests/` that exercises the script without requiring root or
a running Supervisor daemon. The test will verify:

- the script exists and passes `bash -n` when Bash is available;
- help output documents the supported commands and required install parameters;
- a configuration-rendering test mode produces the expected worker command, program name,
  lifecycle settings, and log paths;
- rendered configuration references the environment file but does not embed its contents;
- invalid or missing required paths fail without writing system files;
- no unsupported `uninstall` or direct `nohup` behavior is exposed.

The existing structured worker unit tests remain unchanged because this feature only controls the
external process lifecycle.

## Deployment acceptance

On the Ubuntu host, installation is accepted when:

```bash
sudo supervisorctl status dcagent-structured-worker
```

reports `RUNNING`, the worker logs show its polling loop without configuration errors, and an
existing structured job moves from `queued` with `checkpointRow: 0` to active processing.
