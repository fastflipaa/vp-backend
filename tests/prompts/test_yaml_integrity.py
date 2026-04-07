"""YAML prompt file integrity tests.

Validates all REQUIRED YAML prompt files load without error and have
required fields. The exact file count is no longer enforced because
new prompts get added with each phase (Phase 19 added followup.yaml
with attempt-based prompts, Phase 20 added reengagement_outreach.yaml
and drip_reentry.yaml). The test now verifies the REQUIRED set is
present rather than locking the count.
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
    """Verify the REQUIRED YAML files exist (not exact count).

    New prompts may be added with each phase. The test enforces a
    minimum required set rather than an exact count so that adding
    a new prompt doesn't require updating this test (which is what
    caused the test to silently rot through Phases 19 and 20).
    """

    REQUIRED_V1_FILES = {
        "classification.yaml",
        "followup.yaml",       # Phase 19
        "greeting.yaml",
        "handoff.yaml",
        "human_reentry.yaml",
        "qualifying.yaml",
        "recovery.yaml",
        "reengagement.yaml",
        "scheduling.yaml",
    }

    def test_required_v1_files_present(self):
        """All historically-required v1/ YAML files exist."""
        present = {f.name for f in V1_DIR.glob("*.yaml")}
        missing = self.REQUIRED_V1_FILES - present
        assert not missing, f"Missing required v1/ files: {missing}"

    def test_fallback_exists_at_root(self):
        """fallback.yaml exists at the levitas/ root (not in v1/)."""
        assert (LEVITAS_DIR / "fallback.yaml").exists()

    def test_v1_has_at_least_required_count(self):
        """v1/ has at least the required number of files (more is fine)."""
        v1_files = list(V1_DIR.glob("*.yaml"))
        assert len(v1_files) >= len(self.REQUIRED_V1_FILES), (
            f"v1/ has {len(v1_files)} files, expected >= {len(self.REQUIRED_V1_FILES)}"
        )


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
            "greeting.yaml",
            "handoff.yaml",
            "qualifying.yaml",
            "recovery.yaml",
            "scheduling.yaml",
        ],
    )
    def test_state_specific_yaml_has_system_prompt(self, yaml_name: str):
        """Single-prompt state YAMLs have a system_prompt field.

        followup.yaml is excluded because Phase 19 introduced multi-attempt
        prompts (system_prompt_attempt_1/2/3) -- see test_followup_has_attempt_prompts.
        reengagement.yaml is excluded because it uses tiered prompts
        (system_prompt_hot/warm/cold) -- see test_reengagement_has_tiered_prompts.
        """
        path = V1_DIR / yaml_name
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "system_prompt" in data, (
            f"{yaml_name} missing 'system_prompt' field. Keys: {list(data.keys())}"
        )

    def test_followup_has_attempt_prompts(self):
        """followup.yaml uses system_prompt_attempt_1/2/3 (Phase 19 schema)."""
        path = V1_DIR / "followup.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for key in ("system_prompt_attempt_1", "system_prompt_attempt_2", "system_prompt_attempt_3"):
            assert key in data, f"followup.yaml missing '{key}'"

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
        """human_reentry.yaml has content_es, content_en, and version field."""
        path = V1_DIR / "human_reentry.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "content_es" in data, "human_reentry.yaml missing content_es"
        assert "content_en" in data, "human_reentry.yaml missing content_en"
        # Plan 18-03 standardized: all v1 YAMLs now have version field
        assert "version" in data, "human_reentry.yaml missing version field"
        assert data["version"] == "1.0"


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
