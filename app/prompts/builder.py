"""PromptBuilder — loads YAML prompt configs and renders via Jinja2.

Each LEVITAS prompt is stored as a versioned YAML file with Jinja2 templates.
v1/ files are FROZEN (ported character-for-character from n8n). Improvements
go in v2/.

Usage:
    pb = PromptBuilder("v1")
    system_prompt, user_msg = pb.render("greeting", context_dict)
"""

from __future__ import annotations

from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound, UndefinedError
import structlog

logger = structlog.get_logger()

PROMPTS_DIR = Path(__file__).parent / "levitas"

_jinja_env = Environment(
    loader=FileSystemLoader(str(PROMPTS_DIR)),
    undefined=StrictUndefined,  # Catch missing template vars at render time
    autoescape=False,           # Prompts are plain text, not HTML
)


class PromptBuilder:
    """Loads YAML prompt configs and renders Jinja2 templates with lead context."""

    def __init__(self, version: str = "v1") -> None:
        self.version = version

    def render(self, state: str, context: dict) -> tuple[str, str]:
        """Render a state prompt with lead context.

        Args:
            state: Pipeline state name (e.g. "greeting", "qualifying").
            context: Dict with lead data (lead_name, building, language, etc.).

        Returns:
            Tuple of (rendered_system_prompt, rendered_user_message).

        Raises:
            FileNotFoundError: If the YAML template is not found.
            ValueError: If a required template variable is missing.
        """
        # Language variant selection: try {state}_{lang}.yaml, fallback to {state}.yaml
        language = context.get("language", "es")
        lang_template_path = f"{self.version}/{state.lower()}_{language}.yaml"
        if (PROMPTS_DIR / lang_template_path).exists():
            template_path = lang_template_path
        else:
            template_path = f"{self.version}/{state.lower()}.yaml"

        raw_yaml = self._load_yaml(template_path)

        # Validate frozen flag
        if raw_yaml.get("frozen") is True:
            logger.debug("prompt.frozen", state=state, version=self.version)
        elif self.version == "v1":
            logger.warning(
                "prompt.not_frozen_v1",
                state=state,
                msg="v1 prompt should be frozen — migration contract",
            )

        system_prompt = self._render_template(
            raw_yaml.get("system_prompt", ""), context, f"{state}.system_prompt"
        )

        # Human re-entry injection: prepend context when AI resumes after human agent
        if context.get("human_reentry") and context["human_reentry"].get(
            "had_human_interaction"
        ):
            try:
                reentry_yaml = self._load_yaml("v1/human_reentry.yaml")
                reentry_key = "content_en" if language == "en" else "content_es"
                reentry_text = self._render_template(
                    reentry_yaml.get(reentry_key, ""),
                    context.get("human_reentry", {}),
                    "human_reentry",
                )
                system_prompt = reentry_text + "\n\n" + system_prompt
            except Exception:
                logger.exception(
                    "prompt.human_reentry_injection_failed", state=state
                )

        user_message = self._render_template(
            raw_yaml.get("user_template", ""), context, f"{state}.user_template"
        )

        return system_prompt, user_message

    def render_fallback(self, context: dict) -> tuple[str, str]:
        """Load fallback.yaml and return (fallback_es, fallback_en).

        Fallback messages are used when the circuit breaker is open.
        Not version-prefixed — lives at levitas/fallback.yaml.
        """
        raw_yaml = self._load_yaml("fallback.yaml")
        fallback_es = self._render_template(
            raw_yaml.get("fallback_es", ""), context, "fallback.es"
        )
        fallback_en = self._render_template(
            raw_yaml.get("fallback_en", ""), context, "fallback.en"
        )
        return fallback_es, fallback_en

    def get_config(self, state: str) -> dict:
        """Return the raw YAML config dict for a state.

        Useful for checking model, max_tokens, temperature, or other
        config fields without rendering templates.
        """
        template_path = f"{self.version}/{state.lower()}.yaml"
        return self._load_yaml(template_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_yaml(self, template_path: str) -> dict:
        """Load and parse a YAML template file."""
        full_path = PROMPTS_DIR / template_path
        if not full_path.exists():
            raise FileNotFoundError(
                f"Prompt config not found: {full_path} "
                f"(expected at {template_path})"
            )
        with open(full_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid YAML config at {template_path}: expected dict")
        return raw

    def _render_template(self, template_str: str, context: dict, label: str) -> str:
        """Render a Jinja2 template string with context."""
        if not template_str:
            return ""
        try:
            tmpl = _jinja_env.from_string(str(template_str))
            return tmpl.render(**context)
        except UndefinedError as e:
            logger.error(
                "prompt.render_error",
                label=label,
                missing_var=str(e),
                context_keys=list(context.keys()),
            )
            raise ValueError(
                f"Missing template variable in {label}: {e}"
            ) from e


def render_prompt(state: str, context: dict, version: str = "v1") -> tuple[str, str]:
    """Convenience function — render a prompt without instantiating PromptBuilder."""
    return PromptBuilder(version).render(state, context)
