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
            fields = ", ".join(
                sorted(str(field) for field in unexpected_artifact_fields)
            )
            raise ValueError(f"artifact has unexpected fields: {fields}")

        for field in ARTIFACT_FIELDS:
            value = artifact.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"artifact is missing {field}")

        sha256 = artifact["sha256"]
        if not SHA256_PATTERN.fullmatch(sha256):
            raise ValueError(
                "artifact sha256 must be exactly 64 lowercase hexadecimal characters"
            )

        local_path = artifact["localPath"].strip()
        if local_path.casefold().startswith(("http://", "https://")):
            raise ValueError("offline artifact paths must be local")
