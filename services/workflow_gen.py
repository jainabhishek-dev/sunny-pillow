"""Gemini-powered checkpoint generation for new workflows."""

import json
import os

from services.ai_client import load_model_config

# ── Schema + prompt ────────────────────────────────────────────────────────────

_WORKFLOW_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "checkpoints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category":     {"type": "string"},
                    "instructions": {"type": "string"},
                    "type":  {"type": "string", "enum": ["rule", "judgment"]},
                    "scope": {"type": "string", "enum": ["page", "document"]},
                },
                "required": ["category", "instructions", "type", "scope"],
            },
        },
    },
    "required": ["checkpoints"],
}

_GENERATION_PROMPT_TEMPLATE = """\
You are designing a document review workflow for CheckPoint, an AI tool that \
checks educational textbook pages one at a time using vision AI.

Workflow name: {name}
Checkpoint generation notes: {description}

Generate a set of checkpoints for this workflow.

Checkpoints must be specific enough for an AI to apply visually to a single page image.
Good: "Check that all exercise numbers are sequential and flag any gaps or repeats"
Bad: "Check for errors" or "Review the content"\
"""


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_workflow_content(name: str, ai_notes: str) -> dict:
    """
    Generate checkpoints for a new workflow using Gemini structured output.

    The reviewer persona is constructed from the workflow name at runtime, so
    only checkpoints are generated here.

    Always uses GEMINI_API_KEY and the model from model_config.yaml,
    regardless of the active provider setting (structured output with
    response_schema is a Gemini-only feature).

    Args:
        name: workflow display name (e.g. "HSE")
        ai_notes: admin's detailed notes on what categories and checks to
                  generate — passed directly to Gemini as generation context

    Returns:
        {"checkpoints": [{"category", "instructions", "type", "scope"}, ...]}

    Raises:
        RuntimeError: if GEMINI_API_KEY is not set.
    """
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "It is required for AI workflow generation."
        )

    config = load_model_config()

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=config["model"],
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            response_mime_type="application/json",
            response_schema=_WORKFLOW_GENERATION_SCHEMA,
        ),
    )

    prompt = _GENERATION_PROMPT_TEMPLATE.format(name=name, description=ai_notes)
    response = model.generate_content(prompt)
    return json.loads(response.text)
