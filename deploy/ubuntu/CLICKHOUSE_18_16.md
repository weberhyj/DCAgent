# Ubuntu ClickHouse 18.16 deployment

This runbook covers the apt-installed ClickHouse 18.16.1 deployment used by the
non-Docker Ubuntu host. It uses the legacy users XML model and the explicit
`legacy_18_16` application profile. The modern SQL-RBAC
`deploy/offline/clickhouse-init.sh` is not used on ClickHouse 18.16.

## Before changing the host

Run the version and service checks as an operator. Privileged commands are marked
with `sudo`; review every target before running it.

```bash
apt-cache policy clickhouse-server
clickhouse-server --version
sudo systemctl status clickhouse-server --no-pager
```

Continue only when the server reports `18.16.x` (the supported target is 18.16.1).
Record the current service state and make a backup before editing configuration:

```bash
sudo cp --preserve=mode,ownership,timestamps \
  /etc/clickhouse-server/users.xml \
  /etc/clickhouse-server/users.xml.bak.$(date +%Y%m%d%H%M%S)
```

The repository does not overwrite `/etc/clickhouse-server/users.xml`. Open that
file with `sudoedit`, review the example at
`deploy/ubuntu/clickhouse-18.16-users.xml.example`, and manually merge only the
two user entries inside its existing `<users>` element. Keep the operator's
existing profiles, quotas, and includes. Restrict `<networks>` to the actual
loopback/private subnet(s); never add a public or `0.0.0.0/0` network.

## Generate password hashes

Generate a SHA-256 digest without echoing the password into shell history. Run
once per account and paste only the lowercase 64-character digest into the
corresponding `<password_sha256_hex>` element during the reviewed merge:

```bash
read -r -s password; printf '\n'
printf %s "$password" | sha256sum
unset password
```

Do not commit the digest or plaintext password to this repository. The query user
must remain read-only (`readonly=1`); the ingest user is writable only within the
allow-listed `default` database under the legacy access model.

After saving the manually reviewed XML, validate and restart the service. A
restart is privileged and interrupts active ClickHouse work, so schedule it:

```bash
python3 -c 'import xml.etree.ElementTree as ET; ET.parse("/etc/clickhouse-server/users.xml")'
sudo systemctl restart clickhouse-server
sudo systemctl status clickhouse-server --no-pager
```

## Connection and capability probes

Use password prompts or protected files; do not put secrets in command arguments.
Probe both accounts and confirm the server version before restarting application
processes:

```bash
clickhouse-client --host 127.0.0.1 --user dc_agent_query --password \
  --query 'SELECT version()'
clickhouse-client --host 127.0.0.1 --user dc_agent_query --password \
  --query 'SELECT 1'
clickhouse-client --host 127.0.0.1 --user dc_agent_ingest --password \
  --query 'SELECT 1'
```

For an operator-reviewed write probe, create a temporary table in `default`,
insert one row, select it with the query account, and drop the table afterwards.
Treat the create/drop commands as destructive and run them only in the approved
database. The ingest account must not be able to read or write other databases.

## API and worker environment

Set the following in both the API and structured-worker Supervisor environment
files. Keep the existing role-specific password-file paths protected and outside
the repository:

```dotenv
CLICKHOUSE_URL=http://127.0.0.1:8123
CLICKHOUSE_COMPATIBILITY_MODE=legacy_18_16
CLICKHOUSE_QUERY_USER=dc_agent_query
CLICKHOUSE_QUERY_PASSWORD_FILE=/etc/dc-agent/secrets/clickhouse-query-password
CLICKHOUSE_INGEST_USER=dc_agent_ingest
CLICKHOUSE_INGEST_PASSWORD_FILE=/etc/dc-agent/secrets/clickhouse-ingest-password
```

The API uses the query account; the worker uses the ingest account. Do not set
`CLICKHOUSE_COMPATIBILITY_MODE=modern` on this host. The legacy preflight checks
`SELECT version()` and required query settings before a worker claims a job.

Restart and inspect Supervisor using the repository's manager (or its equivalent
installation already present on the host):

```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl restart dcagent-api
sudo supervisorctl restart dcagent-structured-worker
sudo supervisorctl status dcagent-api dcagent-structured-worker
```

The worker should remain `RUNNING`. Poll the existing structured-status endpoint
for the affected source (replace `{source_id}` with the real identifier) and
confirm the queued job advances without a new upload:

```bash
curl --fail --silent --show-error \
  http://127.0.0.1:8000/api/knowledge/sources/{source_id}/structured-status
```

`attempt` and `checkpointRow` should become observable as batches complete. A
preflight failure occurs before claim, leaving the job queued with `attempt=0`;
after correcting the environment, restarting the worker recovers that queued job
without requiring an XLSX re-upload.

## Rollback

If preflight or publication fails, stop the worker and preserve its logs. Restore
the previous application build and set its previous environment values, then
restart the API and worker through Supervisor. If the XML change caused the
failure, stop ClickHouse, restore the reviewed `.bak.<timestamp>` backup, validate
the config, and restart it. Existing active publications remain available and
unclaimed queued jobs remain recoverable; do not delete staging data or rewrite
system configuration without operator review.
