"""YAML prompt file integrity tests.

Validates all YAML prompt files load without error, have required fields,
and the expected file count is correct.

v1/ has 9 YAML files + 1 fallback.yaml at the levitas/ root = 10 total.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest
import yaml

# Resolve the prompts directory from the source tree
LEVITAS_DIR = Path(__file__).parent.parent.parent / "app" / "prompts" / "levitas"
V1_DIR = LEVITAS_DIR / "v1"


class TestYamlFileCount:
    """Verify expected number of YAML files exist."""

    def test_v1_has_exactly_9_files(self):
        """v1/ directory has exactly 9 YAML prompt files."""
        yaml_files = list(V1_DIR.glob("*.yaml"))
        assert len(yaml_files) == 9, (
            f"Expected 9 YAML files in v1/, found {len(yaml_files)}: "
            f"{[f.name for f in yaml_files]}"
        )

    def test_fallback_exists_at_root(self):
        """fallback.yaml exists at the levitas/ root (not in v1/)."""
        assert (LEVITAS_DIR / "fallback.yaml").exists()

    def test_total_yaml_count_is_10(self):
        """Total YAML files (v1/ + root) = 10."""
        v1_files = list(V1_DIR.glob("*.yaml"))
        root_files = list(LEVITAS_DIR.glob("*.yaml"))
        total = len(v1_files) + len(root_files)
        assert total == 10, f"Expected 10 total, found {total}"


class TestV1YamlIntegrity:
    """Each v1/ YAML file loads and has required fields."""

    @pytest.fixture()
    def v1_yaml_files(self) -> list[Path]:
        """List all v1 YAML files."""
        return sorted(V1_DIR.glob("*.yaml"))

    @pytest.mark.parametrize(
        "yaml_name",
        [
            "classification.yaml",
            "followup.yaml",
            "greeting.yaml",
            "handoff.yaml",
            "human_reentry.yaml",
            "qualifying.yaml",
            "recovery.yaml",
            "reengagement.yaml",
            "scheduling.yaml",
        ],
    )
    def test_yaml_loads_without_error(self, yaml_name: str):
        """Each v1 YAML file loads via yaml.safe_load without error."""
        path = V1_DIR / yaml_name
        assert path.exists(), f"Expected file not found: {yaml_name}"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"{yaml_name} did not load as dict"

    @pytest.mark.parametrize(
        "yaml_name",
        [
            "classification.yaml",
            "followup.yaml",
            "greeting.yaml",
            "handoff.yaml",
            "qualifying.yaml",
            "recovery.yaml",
            "scheduling.yaml",
        ],
    )
    def test_state_specific_yaml_has_system_prompt(self, yaml_name: str):
        """State-specific YAMLs have a system_prompt field."""
        path = V1_DIR / yaml_name
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "system_prompt" in data, (
            f"{yaml_name} missing 'system_prompt' field. Keys: {list(data.keys())}"
        )

    def test_reengagement_has_tiered_prompts(self):
        """reengagement.yaml uses system_prompt_hot/warm/cold (not system_prompt)."""
        path = V1_DIR / "reengagement.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for key in ("system_prompt_hot", "system_prompt_warm", "system_prompt_cold"):
            assert key in data, f"reengagement.yaml missing '{key}'"

    @pytest.mark.parametrize(
        "yaml_name",
        [
            "classification.yaml",
            "followup.yaml",
            "greeting.yaml",
            "handoff.yaml",
            "qualifying.yaml",
            "recovery.yaml",
            "reengagement.yaml",
            "scheduling.yaml",
        ],
    )
    def test_v1_yaml_has_version_field(self, yaml_name: str):
        """v1 prompt files have a 'version' field (except human_reentry)."""
        path = V1_DIR / yaml_name
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "version" in data, (
            f"{yaml_name} missing 'version' field. Keys: {list(data.keys())}"
        )

    def test_human_reentry_structure(self):
        """human_reentry.yaml has content_es and content_en (no version field)."""
        path = V1_DIR / "human_reentry.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "content_es" in data, "human_reentry.yaml missing content_es"
        assert "content_en" in data, "human_reentry.yaml missing content_en"
        # Document: human_reentry is the only v1 file without a version field
        # Plan 18-03 may add it. Test documents the current state.
        assert "version" not in data, (
            "human_reentry.yaml now has version field -- update this test"
        )


class TestFallbackYaml:
    """fallback.yaml loads correctly with bilingual fallback messages."""

    def test_fallback_loads(self):
        """fallback.yaml loads as a dict."""
        path = LEVITAS_DIR / "fallback.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_fallback_has_es_and_en(self):
        """fallback.yaml has both fallback_es and fallback_en."""
        path = LEVITAS_DIR / "fallback.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "fallback_es" in data
        assert "fallback_en" in data

    def test_fallback_has_version(self):
        """fallback.yaml has a version field."""
        path = LEVITAS_DIR / "fallback.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "version" in data
