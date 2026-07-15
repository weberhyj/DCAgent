from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from app.offline_artifacts import validate_artifact_manifest


REQUIRED_ARTIFACT_FIELDS = {
    "name",
    "kind",
    "version",
    "sha256",
    "license",
    "localPath",
}


def valid_artifact() -> dict[str, str]:
    return {
        "name": "docling-layout-model",
        "kind": "docling-artifact",
        "version": "2.40.0",
        "sha256": "a" * 64,
        "license": "MIT",
        "localPath": "/models/docling/layout",
    }


class OfflineArtifactManifestTest(unittest.TestCase):
    def test_accepts_nonempty_manifest_with_required_local_metadata(self) -> None:
        validate_artifact_manifest({"artifacts": [valid_artifact()]})

    def test_rejects_missing_or_blank_required_fields(self) -> None:
        for field in sorted(REQUIRED_ARTIFACT_FIELDS):
            for value in (None, "", "   "):
                with self.subTest(field=field, value=value):
                    artifact = valid_artifact()
                    if value is None:
                        artifact.pop(field)
                    else:
                        artifact[field] = value

                    with self.assertRaisesRegex(
                        ValueError, rf"^artifact is missing {field}$"
                    ):
                        validate_artifact_manifest({"artifacts": [artifact]})

    def test_rejects_missing_empty_or_non_list_artifacts(self) -> None:
        for payload in ({}, {"artifacts": []}, {"artifacts": {}}, None):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(
                    ValueError, "^artifact manifest must contain a nonempty artifacts list$"
                ):
                    validate_artifact_manifest(payload)  # type: ignore[arg-type]

    def test_rejects_non_object_artifact_entries(self) -> None:
        for artifact in ("model", 1, None, []):
            with self.subTest(artifact=artifact):
                with self.assertRaisesRegex(
                    ValueError, "^artifact entries must be objects$"
                ):
                    validate_artifact_manifest({"artifacts": [artifact]})

    def test_requires_exactly_64_lowercase_hexadecimal_sha256_characters(self) -> None:
        invalid_checksums = (
            "a" * 63,
            "a" * 65,
            "A" * 64,
            "g" * 64,
            "a" * 63 + " ",
        )
        for checksum in invalid_checksums:
            with self.subTest(checksum=checksum):
                artifact = valid_artifact()
                artifact["sha256"] = checksum

                with self.assertRaisesRegex(
                    ValueError,
                    "^artifact sha256 must be exactly 64 lowercase hexadecimal characters$",
                ):
                    validate_artifact_manifest({"artifacts": [artifact]})

    def test_rejects_uri_and_network_share_artifact_paths(self) -> None:
        for local_path in (
            "http://models.example/docling",
            "https://models.example/paddleocr",
            "ftp://models.example/libreoffice",
            "s3://offline-bucket/paddleocr",
            "file://server/share/docling",
            "urn:dc-agent:docling",
            r"\\server\share\docling",
            "//server/share/docling",
        ):
            with self.subTest(local_path=local_path):
                artifact = valid_artifact()
                artifact["localPath"] = local_path

                with self.assertRaisesRegex(
                    ValueError,
                    "^offline artifact paths must be local filesystem paths; "
                    "network shares and URI schemes are not allowed$",
                ):
                    validate_artifact_manifest({"artifacts": [artifact]})

    def test_accepts_posix_relative_and_windows_drive_root_artifact_paths(self) -> None:
        for local_path in (
            "/models/docling",
            "models/docling",
            r"C:\models\docling",
            "C:/models/docling",
        ):
            with self.subTest(local_path=local_path):
                artifact = valid_artifact()
                artifact["localPath"] = local_path

                validate_artifact_manifest({"artifacts": [artifact]})

    def test_rejects_properties_outside_the_locked_contract(self) -> None:
        artifact = valid_artifact()
        artifact["downloadUrl"] = "https://models.example/docling"
        with self.assertRaisesRegex(
            ValueError, "^artifact has unexpected fields: downloadUrl$"
        ):
            validate_artifact_manifest({"artifacts": [artifact]})

        with self.assertRaisesRegex(
            ValueError, "^artifact manifest has unexpected fields: mirrorUrl$"
        ):
            validate_artifact_manifest(
                {
                    "artifacts": [valid_artifact()],
                    "mirrorUrl": "https://models.example",
                }
            )

    def test_json_schema_matches_the_runtime_manifest_contract(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "deploy"
            / "offline"
            / "artifacts.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        artifact_schema = schema["$defs"]["artifact"]

        self.assertEqual(schema["required"], ["artifacts"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(artifact_schema["required"]), REQUIRED_ARTIFACT_FIELDS)
        self.assertFalse(artifact_schema["additionalProperties"])
        self.assertEqual(schema["properties"]["artifacts"]["minItems"], 1)

        properties = artifact_schema["properties"]
        for field in REQUIRED_ARTIFACT_FIELDS:
            with self.subTest(field=field):
                self.assertEqual(properties[field]["type"], "string")

        checksum_pattern = properties["sha256"]["pattern"]
        self.assertIsNotNone(re.fullmatch(checksum_pattern, "a" * 64))
        self.assertIsNone(re.fullmatch(checksum_pattern, "A" * 64))

        local_path_pattern = properties["localPath"]["pattern"]
        for local_path in (
            "/models/docling",
            "models/docling",
            r"C:\models\docling",
            "C:/models/docling",
        ):
            with self.subTest(schema_local_path=local_path):
                self.assertIsNotNone(re.fullmatch(local_path_pattern, local_path))

        for remote_path in (
            "https://models.example/docling",
            "ftp://models.example/docling",
            "s3://offline-bucket/docling",
            "file://server/share/docling",
            "urn:dc-agent:docling",
            r"\\server\share\docling",
            "//server/share/docling",
        ):
            with self.subTest(schema_remote_path=remote_path):
                self.assertIsNone(re.fullmatch(local_path_pattern, remote_path))

        description = artifact_schema["description"].lower()
        for artifact_family in (
            "docling",
            "paddleocr detection",
            "paddleocr recognition",
            "paddleocr classification",
            "paddlepaddle cpu wheels",
            "libreoffice",
            "poppler",
        ):
            with self.subTest(artifact_family=artifact_family):
                self.assertIn(artifact_family, description)


if __name__ == "__main__":
    unittest.main()
