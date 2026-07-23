#!/bin/sh
set -eu

query_user=${CLICKHOUSE_QUERY_USER:?CLICKHOUSE_QUERY_USER is required}
ingest_user=${CLICKHOUSE_INGEST_USER:?CLICKHOUSE_INGEST_USER is required}

case "$query_user" in
  ''|*[!A-Za-z0-9_]*|[0-9]*)
    echo "CLICKHOUSE_QUERY_USER must be a safe ClickHouse identifier" >&2
    exit 1
    ;;
esac
case "$ingest_user" in
  ''|*[!A-Za-z0-9_]*|[0-9]*)
    echo "CLICKHOUSE_INGEST_USER must be a safe ClickHouse identifier" >&2
    exit 1
    ;;
esac
if [ "$query_user" = "$ingest_user" ]; then
  echo "ClickHouse query and ingestion users must be distinct" >&2
  exit 1
fi

query_password=$(cat /run/secrets/clickhouse_query_password)
ingest_password=$(cat /run/secrets/clickhouse_ingest_password)
case "$query_password" in
  ???????????????????????????????????????????) ;;
  *) echo "ClickHouse query password has an invalid format" >&2; exit 1 ;;
esac
case "$query_password" in
  *[!A-Za-z0-9_-]*) echo "ClickHouse query password has an invalid format" >&2; exit 1 ;;
esac
case "$ingest_password" in
  ???????????????????????????????????????????) ;;
  *) echo "ClickHouse ingestion password has an invalid format" >&2; exit 1 ;;
esac
case "$ingest_password" in
  *[!A-Za-z0-9_-]*) echo "ClickHouse ingestion password has an invalid format" >&2; exit 1 ;;
esac

clickhouse-client --multiquery <<SQL
CREATE USER IF NOT EXISTS \`$query_user\` IDENTIFIED WITH sha256_password BY '$query_password';
ALTER USER \`$query_user\` IDENTIFIED WITH sha256_password BY '$query_password';
REVOKE ALL ON *.* FROM \`$query_user\`;
GRANT SELECT ON default.* TO \`$query_user\`;

CREATE USER IF NOT EXISTS \`$ingest_user\` IDENTIFIED WITH sha256_password BY '$ingest_password';
ALTER USER \`$ingest_user\` IDENTIFIED WITH sha256_password BY '$ingest_password';
REVOKE ALL ON *.* FROM \`$ingest_user\`;
GRANT CREATE TABLE, INSERT, SELECT, ALTER TABLE, DROP TABLE, TRUNCATE ON default.* TO \`$ingest_user\`;
GRANT SELECT ON system.tables TO \`$ingest_user\`;
SQL
