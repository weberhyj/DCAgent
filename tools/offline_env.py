from __future__ import annotations

import argparse
import base64
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path


ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
NUMERIC_IDENTITY = re.compile(r"^(?:0|[1-9][0-9]*)$")
PASSWORD = re.compile(r"^[A-Za-z0-9_-]{43}$")
MANAGED_SECRET_NAMES = (
    "postgres-password",
    "database-url",
    "clickhouse-query-password",
    "clickhouse-ingest-password",
)
MANAGED_SECRET_ENV = {
    "POSTGRES_PASSWORD_FILE": "postgres-password",
    "DATABASE_URL_SECRET_FILE": "database-url",
    "CLICKHOUSE_QUERY_PASSWORD_FILE": "clickhouse-query-password",
    "CLICKHOUSE_INGEST_PASSWORD_FILE": "clickhouse-ingest-password",
}
POSTGRES_SECRET_NAMES = ("postgres-password", "database-url")
CLICKHOUSE_SECRET_NAMES = (
    "clickhouse-query-password",
    "clickhouse-ingest-password",
)
CLICKHOUSE_ENV_DEFAULTS = {
    "CLICKHOUSE_QUERY_PASSWORD_FILE": (
        "../../artifacts/secrets/clickhouse-query-password"
    ),
    "CLICKHOUSE_INGEST_PASSWORD_FILE": (
        "../../artifacts/secrets/clickhouse-ingest-password"
    ),
}


class DeploymentError(RuntimeError):
    pass


def _load_env_text(text: str) -> OrderedDict[str, str]:
    values: OrderedDict[str, str] = OrderedDict()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if ENV_KEY.fullmatch(key) is None:
            continue
        if key in values:
            raise DeploymentError(
                f"Offline environment key {key} must appear exactly once"
            )
        values[key] = value.strip()
    return values


def load_env(path: Path) -> OrderedDict[str, str]:
    if not path.is_file():
        return OrderedDict()
    return _load_env_text(path.read_text(encoding="utf-8"))


def _render_env_values(text: str, updates: Mapping[str, str]) -> str:
    lines = text.splitlines()
    for name, value in updates.items():
        pattern = re.compile(rf"^\s*{re.escape(name)}\s*=")
        indexes = [index for index, line in enumerate(lines) if pattern.match(line)]
        if len(indexes) > 1:
            raise DeploymentError(
                f"Offline environment key {name} must appear exactly once"
            )
        replacement = f"{name}={value}"
        if indexes:
            lines[indexes[0]] = replacement
        else:
            lines.append(replacement)
    return "\n".join(lines) + "\n"


def _atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding=encoding,
            newline="\n",
        ) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def set_env_value(path: Path, name: str, value: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    _atomic_write_text(path, _render_env_values(text, {name: value}))


def canonical_numeric_identity(
    name: str,
    value: str,
    *,
    reject_root: bool,
) -> str:
    if NUMERIC_IDENTITY.fullmatch(value) is None:
        raise DeploymentError(f"{name} must use canonical decimal notation")
    number = int(value)
    if number > 2_147_483_647 or (reject_root and number == 0):
        raise DeploymentError(f"{name} is outside the supported range")
    return str(number)


def _assert_no_symbolic_link_ancestors(path: Path, name: str) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise DeploymentError(
                f"{name} path ancestors must not be symbolic links: {current}"
            )
        if current.parent == current:
            break
        current = current.parent


def resolve_env_path(
    env_path: Path,
    name: str,
    raw_value: str,
    *,
    environ: Mapping[str, str],
) -> Path:
    if not raw_value or "'" in raw_value or '"' in raw_value:
        raise DeploymentError(f"{name} must use one unquoted direct path")
    variable = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", raw_value)
    if variable is not None:
        raw_value = environ.get(variable.group(1), "")
    elif "$" in raw_value:
        raise DeploymentError(f"{name} uses unresolved variable syntax")
    if not raw_value:
        raise DeploymentError(f"{name} must resolve to a non-empty path")
    candidate = Path(raw_value)
    unresolved = candidate if candidate.is_absolute() else env_path.parent / candidate
    _assert_no_symbolic_link_ancestors(unresolved, name)
    return unresolved.resolve(strict=False)


def _current_identity() -> tuple[str, str]:
    uid = subprocess.run(
        ["id", "-u"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    gid = subprocess.run(
        ["id", "-g"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return uid, gid


def _assert_local_deployment_environment(environ: Mapping[str, str]) -> None:
    docker_host = environ.get("DOCKER_HOST", "")
    if docker_host and docker_host != "unix:///var/run/docker.sock":
        raise DeploymentError(
            "Only local rootful Docker at /var/run/docker.sock is supported"
        )
    docker_context = environ.get("DOCKER_CONTEXT", "")
    if docker_context and docker_context != "default":
        raise DeploymentError("Only the local default Docker context is supported")


def _resolve_with_override(
    env_path: Path,
    name: str,
    value: str,
    *,
    environ: Mapping[str, str],
) -> Path:
    configured = resolve_env_path(
        env_path,
        name,
        value,
        environ=environ,
    )
    override = environ.get(name)
    if override is not None:
        override_path = resolve_env_path(
            env_path,
            name,
            override,
            environ=environ,
        )
        if override_path != configured:
            raise DeploymentError(
                f"{name} shell override must resolve to the same path as deploy/offline/.env"
            )
    return configured


def _managed_paths(repo_root: Path) -> dict[str, Path]:
    secret_dir = repo_root / "artifacts" / "secrets"
    return {name: secret_dir / name for name in MANAGED_SECRET_NAMES}


def _assert_managed_secret_paths(
    repo_root: Path,
    env_path: Path,
    values: Mapping[str, str],
    *,
    environ: Mapping[str, str],
) -> dict[str, Path]:
    paths = _managed_paths(repo_root)
    for env_name, secret_name in MANAGED_SECRET_ENV.items():
        raw_value = values.get(env_name)
        if raw_value is None:
            raise DeploymentError(f"{env_name} must be explicitly defined")
        configured = _resolve_with_override(
            env_path,
            env_name,
            raw_value,
            environ=environ,
        )
        expected = paths[secret_name].resolve(strict=False)
        if configured != expected:
            raise DeploymentError(
                f"{env_name} must use the repository-managed secret path"
            )
    return paths


def _new_password() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")


def _write_secret(path: Path, value: str) -> None:
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise DeploymentError(f"Offline secret must be a regular file: {path}")
    _atomic_write_text(path, value, encoding="ascii", mode=0o600)


def _replace_secret(source: Path, target: Path) -> None:
    os.replace(source, target)


def _publish_secret_set(paths: Mapping[str, Path]) -> None:
    names = set(paths)
    values: dict[str, str] = {}
    if names.intersection(POSTGRES_SECRET_NAMES):
        if not names.issuperset(POSTGRES_SECRET_NAMES):
            raise DeploymentError(
                "Managed offline PostgreSQL secrets must exist as one complete set"
            )
        postgres_password = _new_password()
        values.update(
            {
                "postgres-password": postgres_password,
                "database-url": (
                    "postgresql+psycopg://dc_agent:"
                    f"{postgres_password}@postgres:5432/dc_agent"
                ),
            }
        )
    if names.intersection(CLICKHOUSE_SECRET_NAMES):
        if not names.issuperset(CLICKHOUSE_SECRET_NAMES):
            raise DeploymentError("ClickHouse password files must exist together")
        values.update(
            {
                "clickhouse-query-password": _new_password(),
                "clickhouse-ingest-password": _new_password(),
            }
        )
    if names != set(values):
        raise DeploymentError("Unknown managed offline secret set")

    secret_dir = next(iter(paths.values())).parent
    staging_dir = Path(tempfile.mkdtemp(prefix=".secret-stage-", dir=secret_dir))
    backup_dir = Path(tempfile.mkdtemp(prefix=".secret-backup-", dir=secret_dir))
    backed_up: list[str] = []
    published: list[str] = []
    transaction_finished = False
    try:
        for name, value in values.items():
            _write_secret(staging_dir / name, value)
        staging_paths = {name: staging_dir / name for name in paths}
        _validate_secret_set(staging_paths)
        for name, target in paths.items():
            if target.exists():
                if target.is_symlink() or not target.is_file():
                    raise DeploymentError(
                        f"Offline secret must be a regular file: {target}"
                    )
                _replace_secret(target, backup_dir / name)
                backed_up.append(name)
        for name, target in paths.items():
            _replace_secret(staging_dir / name, target)
            published.append(name)
        _validate_secret_set(paths)
        transaction_finished = True
    except BaseException as publication_error:
        try:
            for name in reversed(published):
                paths[name].unlink(missing_ok=True)
            for name in reversed(backed_up):
                restore_source = backup_dir / f".restore-{name}"
                shutil.copy2(backup_dir / name, restore_source)
                _replace_secret(restore_source, paths[name])
            if backed_up:
                _validate_secret_set(paths)
            transaction_finished = True
        except BaseException as rollback_error:
            raise DeploymentError(
                "Secret publication failed and rollback could not be completed; "
                f"backup retained at {backup_dir}"
            ) from rollback_error
        raise publication_error
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        if transaction_finished:
            shutil.rmtree(backup_dir, ignore_errors=True)


def _assert_regular_non_link(path: Path, context: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise DeploymentError(f"{context} must be one regular non-link file: {path}")


def _assert_directory_non_link(path: Path, context: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise DeploymentError(f"{context} must be one non-link directory: {path}")


def _assert_posix_metadata(
    path: Path,
    *,
    uid: int,
    gid: int,
    mode: int,
    context: str,
) -> None:
    metadata = path.stat()
    if (
        metadata.st_uid != uid
        or metadata.st_gid != gid
        or metadata.st_mode & 0o777 != mode
    ):
        raise DeploymentError(f"{context} owner or mode is unsafe: {path}")


def _assert_posix_owner(
    path: Path,
    *,
    uid: int,
    gid: int,
    context: str,
) -> None:
    _assert_posix_metadata(
        path,
        uid=uid,
        gid=gid,
        mode=stat.S_IMODE(path.stat().st_mode),
        context=context,
    )


def _validate_postgres_secret_pair(paths: Mapping[str, Path]) -> None:
    postgres_password = paths["postgres-password"].read_text(encoding="ascii")
    if PASSWORD.fullmatch(postgres_password) is None:
        raise DeploymentError("Offline password has an invalid format")
    database_url = paths["database-url"].read_text(encoding="ascii")
    expected_database_url = (
        f"postgresql+psycopg://dc_agent:{postgres_password}@postgres:5432/dc_agent"
    )
    if database_url != expected_database_url:
        raise DeploymentError(
            "Offline database URL must match the managed PostgreSQL password"
        )


def _validate_clickhouse_secret_pair(paths: Mapping[str, Path]) -> None:
    query_password = paths["clickhouse-query-password"].read_text(encoding="ascii")
    ingest_password = paths["clickhouse-ingest-password"].read_text(encoding="ascii")
    for password in (query_password, ingest_password):
        if PASSWORD.fullmatch(password) is None:
            raise DeploymentError("Offline password has an invalid format")
    if query_password == ingest_password:
        raise DeploymentError("ClickHouse query and ingest passwords must differ")


def _validate_secret_set(paths: Mapping[str, Path]) -> None:
    names = set(paths)
    for path in paths.values():
        _assert_regular_non_link(path, "Offline secret")
    if names.intersection(POSTGRES_SECRET_NAMES):
        if not names.issuperset(POSTGRES_SECRET_NAMES):
            raise DeploymentError(
                "Managed offline PostgreSQL secrets must exist as one complete set"
            )
        _validate_postgres_secret_pair(paths)
    if names.intersection(CLICKHOUSE_SECRET_NAMES):
        if not names.issuperset(CLICKHOUSE_SECRET_NAMES):
            raise DeploymentError("ClickHouse password files must exist together")
        _validate_clickhouse_secret_pair(paths)
    if names not in (
        set(POSTGRES_SECRET_NAMES),
        set(CLICKHOUSE_SECRET_NAMES),
        set(MANAGED_SECRET_NAMES),
    ):
        raise DeploymentError("Unknown managed offline secret set")


def prepare_environment(
    repo_root: Path,
    *,
    rotate_secrets: bool = False,
    environ: Mapping[str, str] | None = None,
    verify_posix_metadata: bool = True,
) -> None:
    repo_root = repo_root.resolve()
    effective_environ = os.environ if environ is None else environ
    _assert_local_deployment_environment(effective_environ)

    env_example = repo_root / "deploy" / "offline" / ".env.example"
    env_path = repo_root / "deploy" / "offline" / ".env"
    if not env_example.is_file():
        raise DeploymentError(f"Offline environment example is missing: {env_example}")
    _assert_no_symbolic_link_ancestors(env_path, "deploy/offline/.env")
    created_env = not env_path.exists()
    source_env = env_example if created_env else env_path
    _assert_regular_non_link(source_env, "Offline environment")
    env_text = source_env.read_text(encoding="utf-8")

    raw_uid, raw_gid = _current_identity()
    uid = canonical_numeric_identity("current Linux UID", raw_uid, reject_root=True)
    gid = canonical_numeric_identity("current Linux GID", raw_gid, reject_root=True)
    values = _load_env_text(env_text)
    env_updates: OrderedDict[str, str] = OrderedDict()
    if created_env:
        env_updates["DCAGENT_UID"] = uid
        env_updates["DCAGENT_GID"] = gid
        env_text = _render_env_values(env_text, env_updates)
        values = _load_env_text(env_text)
    for name, expected in (("DCAGENT_UID", uid), ("DCAGENT_GID", gid)):
        actual_value = values.get(name)
        if actual_value is None:
            raise DeploymentError(f"{name} must be explicitly defined")
        actual = canonical_numeric_identity(name, actual_value, reject_root=True)
        if actual != expected:
            raise DeploymentError(
                f"{name} must match the current Linux deployment account"
            )
        override = effective_environ.get(name)
        if (
            override is not None
            and canonical_numeric_identity(
                f"{name} shell override", override, reject_root=True
            )
            != actual
        ):
            raise DeploymentError(
                f"{name} shell override must match deploy/offline/.env"
            )

    for required_name in ("DATA_ROOT", "MODEL_ROOT"):
        if required_name not in values:
            raise DeploymentError(f"{required_name} must be explicitly defined")
    data_root = _resolve_with_override(
        env_path,
        "DATA_ROOT",
        values["DATA_ROOT"],
        environ=effective_environ,
    )
    model_root = _resolve_with_override(
        env_path,
        "MODEL_ROOT",
        values["MODEL_ROOT"],
        environ=effective_environ,
    )
    for name in ("postgres", "clickhouse", "qdrant", "redis"):
        _assert_directory_non_link(
            data_root / name,
            "Vendor bind source",
        )
    _assert_directory_non_link(model_root, "Model bind source")

    # Existing deployments before the ClickHouse pair was introduced are
    # upgraded in memory and committed with one atomic .env replacement.
    clickhouse_keys_present = {name: name in values for name in CLICKHOUSE_ENV_DEFAULTS}
    if not any(clickhouse_keys_present.values()):
        env_updates.update(CLICKHOUSE_ENV_DEFAULTS)
        env_text = _render_env_values(env_text, env_updates)
        values = _load_env_text(env_text)
    elif not all(clickhouse_keys_present.values()):
        raise DeploymentError(
            "Both ClickHouse password file paths must be configured together"
        )

    paths = _assert_managed_secret_paths(
        repo_root,
        env_path,
        values,
        environ=effective_environ,
    )
    secret_dir = paths["postgres-password"].parent

    numeric_uid = int(uid)
    numeric_gid = int(gid)
    writable_paths = [data_root / "raw", data_root / "parquet"]
    for path in writable_paths:
        if path.exists():
            _assert_directory_non_link(path, "Writable bind source")
            if verify_posix_metadata and os.name == "posix":
                _assert_posix_metadata(
                    path,
                    uid=numeric_uid,
                    gid=numeric_gid,
                    mode=path.stat().st_mode & 0o777,
                    context="Writable bind source",
                )

    if secret_dir.exists():
        _assert_directory_non_link(secret_dir, "Offline secret directory")
        if verify_posix_metadata and os.name == "posix":
            _assert_posix_metadata(
                secret_dir,
                uid=numeric_uid,
                gid=numeric_gid,
                mode=secret_dir.stat().st_mode & 0o777,
                context="Offline secret directory",
            )

    postgres_paths = {name: paths[name] for name in POSTGRES_SECRET_NAMES}
    clickhouse_paths = {name: paths[name] for name in CLICKHOUSE_SECRET_NAMES}
    postgres_present = {name for name, path in postgres_paths.items() if path.exists()}
    clickhouse_present = {
        name for name, path in clickhouse_paths.items() if path.exists()
    }
    if postgres_present and postgres_present != set(postgres_paths):
        raise DeploymentError(
            "Managed offline PostgreSQL secrets must exist as one complete set"
        )
    if clickhouse_present and clickhouse_present != set(clickhouse_paths):
        raise DeploymentError("ClickHouse password files must exist together")
    if postgres_present:
        _validate_secret_set(postgres_paths)
    if clickhouse_present:
        _validate_secret_set(clickhouse_paths)
    if verify_posix_metadata and os.name == "posix":
        for name in MANAGED_SECRET_NAMES:
            path = paths[name]
            if path.exists():
                _assert_posix_owner(
                    path,
                    uid=numeric_uid,
                    gid=numeric_gid,
                    context="Offline secret",
                )

    managed_targets = [data_root, model_root, secret_dir, *paths.values()]
    for target in managed_targets:
        _assert_no_symbolic_link_ancestors(target, "Offline managed path")

    # The rotation guard is deliberately the final preflight check before any
    # mkdir/chmod/.env/secret mutation.
    if rotate_secrets:
        pg_version = data_root / "postgres" / "PG_VERSION"
        if pg_version.exists():
            raise DeploymentError(
                "Refusing to rotate initialized PostgreSQL secrets. Stop PostgreSQL, "
                "perform a controlled ALTER ROLE, update both files, restart, and "
                "verify connectivity."
            )

    publish_names: set[str] = set()
    if rotate_secrets:
        publish_names = set(paths)
    else:
        if not postgres_present:
            publish_names.update(POSTGRES_SECRET_NAMES)
        if not clickhouse_present:
            publish_names.update(CLICKHOUSE_SECRET_NAMES)

    created_directories: list[Path] = []
    original_modes: dict[Path, int] = {}

    def secure_directory(path: Path) -> None:
        if path.exists():
            original_modes.setdefault(path, stat.S_IMODE(path.stat().st_mode))
        else:
            path.mkdir(mode=0o700, parents=True)
            created_directories.append(path)
        os.chmod(path, 0o700)

    try:
        for path in writable_paths:
            secure_directory(path)
        secure_directory(secret_dir)
        for path in paths.values():
            if path.exists():
                original_modes.setdefault(path, stat.S_IMODE(path.stat().st_mode))

        if publish_names:
            _publish_secret_set(
                {
                    name: paths[name]
                    for name in MANAGED_SECRET_NAMES
                    if name in publish_names
                }
            )
        _validate_secret_set(paths)
        for path in paths.values():
            os.chmod(path, 0o600)
            if verify_posix_metadata and os.name == "posix":
                _assert_posix_metadata(
                    path,
                    uid=numeric_uid,
                    gid=numeric_gid,
                    mode=0o600,
                    context="Offline secret",
                )

        if created_env or env_updates:
            _atomic_write_text(env_path, env_text)
    except BaseException:
        for path, mode in original_modes.items():
            if path.exists() and not path.is_symlink():
                try:
                    os.chmod(path, mode)
                except OSError:
                    pass
        for path in reversed(created_directories):
            try:
                path.rmdir()
            except OSError:
                pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotate-secrets", action="store_true")
    args = parser.parse_args(argv)
    prepare_environment(
        Path(__file__).resolve().parents[1],
        rotate_secrets=args.rotate_secrets,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DeploymentError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
