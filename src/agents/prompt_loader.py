"""Helper utilities for rendering Jinja2 prompt templates."""
from pathlib import Path
from typing import Any, Dict
from jinja2 import Environment, FileSystemLoader, select_autoescape

PROMPTS_DIR = Path(__file__).parent / "prompts"

_env = Environment(
    loader=FileSystemLoader(str(PROMPTS_DIR)),
    autoescape=select_autoescape(disabled_extensions=("jinja2",)),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_prompt(template_name: str, context: Dict[str, Any]) -> str:
    """Renders a Jinja2 template with the given context."""
    template = _env.get_template(template_name)
    return template.render(**context)

