from __future__ import annotations

import re
from collections.abc import Mapping

ARTIFACT_FIELDS = (
    "name",
    "kind",
    "version",
    "sha256",
    "license",
    "localPath",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
URI_SCHEME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
WINDOWS_DRIVE_ROOT_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")
NETWORK_SHARE_PATTERN = re.compile(r"^[\\/]{2}")


def is_local_filesystem_path(value: str) -> bool:
    candidate = value.strip()
    if not candidate:
        return False
    if WINDOWS_DRIVE_ROOT_PATTERN.match(candidate):
        return True
    return not (URI_SCHEME_PATTERN.match(candidate) or NETWORK_SHARE_PATTERN.match(candidate))


def validate_artifact_manifest(payload: Mapping[str, object]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("artifact manifest must contain a nonempty artifacts list")

    unexpected_manifest_fields = set(payload) - {"artifacts"}
    if unexpected_manifest_fields:
        fields = ", ".join(sorted(str(field) for field in unexpected_manifest_fields))
        raise ValueError(f"artifact manifest has unexpected fields: {fields}")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("artifact manifest must contain a nonempty artifacts list")

    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("artifact entries must be objects")

        unexpected_artifact_fields = set(artifact) - set(ARTIFACT_FIELDS)
        if unexpected_artifact_fields:
            fields = ", ".join(sorted(str(field) for field in unexpected_artifact_fields))
            raise ValueError(f"artifact has unexpected fields: {fields}")

        for field in ARTIFACT_FIELDS:
            value = artifact.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"artifact is missing {field}")

        sha256 = artifact["sha256"]
        if not SHA256_PATTERN.fullmatch(sha256):
            raise ValueError("artifact sha256 must be exactly 64 lowercase hexadecimal characters")

        if not is_local_filesystem_path(artifact["localPath"]):
            raise ValueError(
                "offline artifact paths must be local filesystem paths; "
                "network shares and URI schemes are not allowed"
            )
