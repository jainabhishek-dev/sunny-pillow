"""AI functions for the Review workflow (vision check/review + document check/review)."""

import os
from io import BytesIO

from services.ai_client import load_model_config, parse_response

# ── Vision prompt ─────────────────────────────────────────────────────────────

_VISION_PROMPT_TEMPLATE = (
    'You are a professional content reviewer for LEAD, an educational publishing house'
    ' specialising in "{workflow_name}".\n\n'
    "Review the document page image against ONLY these checkpoints:\n\n"
    "{rules}\n\n"
    "INSTRUCTIONS:\n"
    "- Read the page image carefully.\n"
    "- For each checkpoint, flag EVERY issue you can see.\n"
    "- Use the checkpoint_id shown in brackets for each checkpoint.\n"
    "- Quote exact text from the page (20–80 characters). "
    "For visual issues (images, layout), quote the nearest caption or label instead.\n"
    "- Keep issue and suggestion fields to 15 words or fewer each.\n"
    "- Return a JSON array only. No markdown, no explanation.\n\n"
    'Schema:\n'
    '[{"checkpoint_id": "cp_001", "quote": "...", "location": "Page {page_num}",'
    ' "issue": "...", "suggestion": "..."}]\n\n'
    "Return [] if no violations found on this page."
)

REVIEW_PROMPT = """You are a quality reviewer for LEAD, an educational publishing house.

A vision AI has flagged the following potential errors on this page. Examine the page image carefully and verify each one.

Findings to review:
{findings_list}

For each finding:
- Return "valid" if the quoted text is visible on the page AND the described issue is correct.
- Return "invalid" if the quote cannot be found on the page, or the issue is incorrect or exaggerated.

Return a JSON array only. No markdown, no explanation.
Schema:
[{{"finding_id": 1, "verdict": "valid", "reason": "..."}}]

Keep reason to 10 words or fewer."""

DOCUMENT_PROMPT = """You are a professional reviewer for LEAD, an educational publishing house.

Review the ENTIRE document against ONLY these consistency checks:

{rules}

INSTRUCTIONS:
- Read the full document carefully across all pages.
- For each rule, flag EVERY violation you can find.
- Quote exact text from the document (20–80 characters).
- Include the page number in the location field where possible (e.g. "Page 3").
- Keep issue and suggestion fields to 15 words or fewer each.
- Return a JSON array only. No markdown, no explanation.

Schema:
[{{"checkpoint_id": "cp_052", "quote": "...", "location": "Page X", "issue": "...", "suggestion": "..."}}]

Return [] if no violations found."""

DOCUMENT_REVIEW_PROMPT = """You are a quality reviewer for LEAD, an educational publishing house.

A reviewer has flagged the following potential errors in this document. Examine the document carefully and verify each one.

Findings to review:
{findings_list}

For each finding:
- Return "valid" if the quoted text is present in the document AND the described issue is correct.
- Return "invalid" if the quote cannot be found, or the issue is incorrect or exaggerated.

Return a JSON array only. No markdown, no explanation.
Schema:
[{{"finding_id": 1, "verdict": "valid", "reason": "..."}}]

Keep reason to 10 words or fewer."""


# ── Prompt builders ────────────────────────────────────────────────────────────

def _build_vision_prompt(checkpoints: list[dict], page_num: int, workflow_name: str) -> str:
    rules = "\n".join(
        f"{i + 1}. [{cp['id']}] {cp['instructions'].strip()}"
        for i, cp in enumerate(checkpoints)
    )
    return (
        _VISION_PROMPT_TEMPLATE
        .replace("{workflow_name}", workflow_name)
        .replace("{rules}", rules)
        .replace("{page_num}", str(page_num))
    )


def _build_review_prompt(findings: list[dict]) -> str:
    lines = [
        f'{i + 1}. [finding_id: {f["id"]}] Quote: "{f["quote"]}" | Issue: {f["issue"]}'
        for i, f in enumerate(findings)
    ]
    return REVIEW_PROMPT.format(findings_list="\n".join(lines))


def _build_document_prompt(checkpoints: list[dict]) -> str:
    rules = "\n".join(
        f"{i + 1}. [{cp['id']}] {cp['instructions'].strip()}"
        for i, cp in enumerate(checkpoints)
    )
    return DOCUMENT_PROMPT.format(rules=rules)


def _build_document_review_prompt(findings: list[dict]) -> str:
    lines = [
        f'{i + 1}. [finding_id: {f["id"]}] Quote: "{f["quote"]}" | Issue: {f["issue"]}'
        for i, f in enumerate(findings)
    ]
    return DOCUMENT_REVIEW_PROMPT.format(findings_list="\n".join(lines))


# ── Gemini runners ─────────────────────────────────────────────────────────────

def _run_gemini_vision(image_bytes: bytes, checkpoints: list[dict], page_num: int, config: dict, workflow_name: str) -> list[dict]:
    import google.generativeai as genai
    from PIL import Image

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"API key not found. Set '{config['api_key_env']}'.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=config["model"],
        generation_config=genai.types.GenerationConfig(
            temperature=config["temperature"],
            max_output_tokens=config["max_tokens"],
        ),
    )
    image = Image.open(BytesIO(image_bytes))
    prompt = _build_vision_prompt(checkpoints, page_num, workflow_name)
    response = model.generate_content([prompt, image])
    return parse_response(response.text)


def _run_gemini_review(image_bytes: bytes, prompt: str, config: dict) -> list[dict]:
    import google.generativeai as genai
    from PIL import Image

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"API key not found. Set '{config['api_key_env']}'.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=config["model"],
        generation_config=genai.types.GenerationConfig(
            temperature=config["temperature"],
            max_output_tokens=config["max_tokens"],
        ),
    )
    image = Image.open(BytesIO(image_bytes))
    response = model.generate_content([prompt, image])
    return parse_response(response.text)


def _run_gemini_document(pdf_bytes: bytes, prompt: str, config: dict) -> list[dict]:
    import google.generativeai as genai
    import base64

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"API key not found. Set '{config['api_key_env']}'.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=config["model"],
        generation_config=genai.types.GenerationConfig(
            temperature=config["temperature"],
            max_output_tokens=config["max_tokens"],
        ),
    )
    pdf_data = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    response = model.generate_content([
        {"mime_type": "application/pdf", "data": pdf_data},
        prompt,
    ])
    return parse_response(response.text)


# ── Anthropic runners ──────────────────────────────────────────────────────────

def _run_anthropic_vision(image_bytes: bytes, checkpoints: list[dict], page_num: int, config: dict, workflow_name: str) -> list[dict]:
    import anthropic
    import base64

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"API key not found. Set '{config['api_key_env']}'.")
    client = anthropic.Anthropic(api_key=api_key)
    image_data = base64.standard_b64encode(image_bytes).decode("utf-8")
    prompt = _build_vision_prompt(checkpoints, page_num, workflow_name)
    message = client.messages.create(
        model=config["model"],
        max_tokens=config["max_tokens"],
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
            {"type": "text", "text": prompt},
        ]}],
    )
    return parse_response(message.content[0].text)


def _run_anthropic_review(image_bytes: bytes, prompt: str, config: dict) -> list[dict]:
    import anthropic
    import base64

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"API key not found. Set '{config['api_key_env']}'.")
    client = anthropic.Anthropic(api_key=api_key)
    image_data = base64.standard_b64encode(image_bytes).decode("utf-8")
    message = client.messages.create(
        model=config["model"],
        max_tokens=config["max_tokens"],
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
            {"type": "text", "text": prompt},
        ]}],
    )
    return parse_response(message.content[0].text)


def _run_anthropic_document(pdf_bytes: bytes, prompt: str, config: dict) -> list[dict]:
    import anthropic
    import base64

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"API key not found. Set '{config['api_key_env']}'.")
    client = anthropic.Anthropic(api_key=api_key)
    pdf_data = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    message = client.messages.create(
        model=config["model"],
        max_tokens=config["max_tokens"],
        messages=[{"role": "user", "content": [
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_data}},
            {"type": "text", "text": prompt},
        ]}],
    )
    return parse_response(message.content[0].text)


# ── Public API ─────────────────────────────────────────────────────────────────

def run_vision_check(
    image_bytes: bytes,
    checkpoints: list[dict],
    page_num: int,
    workflow_name: str,
    custom_prompt: str | None = None,
) -> list[dict]:
    """First-pass vision check on a single page image. Returns findings list."""
    config = load_model_config()
    provider = config.get("provider", "gemini")
    if custom_prompt:
        # Substitute page_num placeholder if present, then use as the full prompt
        resolved_prompt = custom_prompt.replace("{page_num}", str(page_num))
        if provider == "gemini":
            import google.generativeai as genai
            from PIL import Image
            api_key = os.getenv(config["api_key_env"])
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=config["model"],
                generation_config=genai.types.GenerationConfig(
                    temperature=config["temperature"],
                    max_output_tokens=config["max_tokens"],
                ),
            )
            image = Image.open(BytesIO(image_bytes))
            response = model.generate_content([resolved_prompt, image])
            findings = parse_response(response.text)
        elif provider == "anthropic":
            import anthropic, base64
            api_key = os.getenv(config["api_key_env"])
            ac = anthropic.Anthropic(api_key=api_key)
            image_data = base64.standard_b64encode(image_bytes).decode("utf-8")
            message = ac.messages.create(
                model=config["model"],
                max_tokens=config["max_tokens"],
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                    {"type": "text", "text": resolved_prompt},
                ]}],
            )
            findings = parse_response(message.content[0].text)
        else:
            findings = []
    elif provider == "gemini":
        findings = _run_gemini_vision(image_bytes, checkpoints, page_num, config, workflow_name)
    elif provider == "anthropic":
        findings = _run_anthropic_vision(image_bytes, checkpoints, page_num, config, workflow_name)
    else:
        raise ValueError(f"Unsupported provider '{provider}'. Supported: gemini, anthropic")
    required = {"checkpoint_id", "quote", "location", "issue", "suggestion"}
    return [f for f in findings if isinstance(f, dict) and required.issubset(f.keys())]


def run_vision_review(image_bytes: bytes, findings: list[dict], page_num: int) -> list[dict]:
    """Second-pass review: validates each finding from the first pass."""
    if not findings:
        return []
    config = load_model_config()
    provider = config.get("provider", "gemini")
    prompt = _build_review_prompt(findings)
    if provider == "gemini":
        reviews = _run_gemini_review(image_bytes, prompt, config)
    elif provider == "anthropic":
        reviews = _run_anthropic_review(image_bytes, prompt, config)
    else:
        return []
    required = {"finding_id", "verdict", "reason"}
    return [r for r in reviews if isinstance(r, dict) and required.issubset(r.keys())]


def run_document_check(pdf_bytes: bytes, checkpoints: list[dict], custom_prompt: str | None = None) -> list[dict]:
    """Document-level check: sends the full PDF with document-scope checkpoints."""
    if not checkpoints and not custom_prompt:
        return []
    config = load_model_config()
    provider = config.get("provider", "gemini")
    prompt = custom_prompt if custom_prompt else _build_document_prompt(checkpoints)
    if provider == "gemini":
        findings = _run_gemini_document(pdf_bytes, prompt, config)
    elif provider == "anthropic":
        findings = _run_anthropic_document(pdf_bytes, prompt, config)
    else:
        return []
    required = {"checkpoint_id", "quote", "location", "issue", "suggestion"}
    return [f for f in findings if isinstance(f, dict) and required.issubset(f.keys())]


def run_document_review(pdf_bytes: bytes, findings: list[dict]) -> list[dict]:
    """Second-pass review for document-level findings using the same PDF."""
    if not findings:
        return []
    config = load_model_config()
    provider = config.get("provider", "gemini")
    prompt = _build_document_review_prompt(findings)
    if provider == "gemini":
        reviews = _run_gemini_document(pdf_bytes, prompt, config)
    elif provider == "anthropic":
        reviews = _run_anthropic_document(pdf_bytes, prompt, config)
    else:
        return []
    required = {"finding_id", "verdict", "reason"}
    return [r for r in reviews if isinstance(r, dict) and required.issubset(r.keys())]
