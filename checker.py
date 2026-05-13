import json
import os
import re
from pathlib import Path
from io import BytesIO

import yaml
from PIL import Image


def _load_model_config() -> dict:
    config_path = Path(__file__).parent / "config" / "model_config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)



EDIT_PROMPT = """You are a professional editorial checker for LEAD, an educational publishing house.

Check the document page image provided against ONLY these style rules:

{rules}

INSTRUCTIONS:
- Read the page image carefully.
- For each rule, flag EVERY violation you can see.
- Quote exact text from the page (20–80 characters).
- Keep issue and suggestion fields to 15 words or fewer each.
- Return a JSON array only. No markdown, no explanation.

Schema:
[{{"checkpoint_id": "cp_001", "quote": "...", "location": "Page {page_num}", "issue": "...", "suggestion": "..."}}]

Return [] if no violations found on this page."""


MATH_PROMPT = """You are a professional mathematics and pedagogy reviewer for LEAD, an educational publishing house.

Review the document page image against ONLY these error categories:

{rules}

INSTRUCTIONS:
- Read the page image carefully.
- For each error category, flag EVERY issue you can see.
- Use the checkpoint_id shown in brackets for each category.
- Quote exact text from the page (20–80 characters).
- Keep issue and suggestion fields to 15 words or fewer each.
- Return a JSON array only. No markdown, no explanation.

Schema:
[{{"checkpoint_id": "cp_037", "quote": "...", "location": "Page {page_num}", "issue": "...", "suggestion": "..."}}]

Return [] if no violations found on this page."""


HSE_PROMPT = """You are a professional content reviewer for LEAD, an educational publishing house specialising in Humans, Society, and Earth (HSE) materials.

Review the document page image against ONLY these error categories:

{rules}

INSTRUCTIONS:
- Read the page image carefully.
- For each error category, flag EVERY issue you can see.
- Use the checkpoint_id shown in brackets for each category.
- Quote exact text from the page (20–80 characters). For visual issues (images, layout), quote the nearest caption or label instead.
- Keep issue and suggestion fields to 15 words or fewer each.
- Return a JSON array only. No markdown, no explanation.

Schema:
[{{"checkpoint_id": "cp_064", "quote": "...", "location": "Page {page_num}", "issue": "...", "suggestion": "..."}}]

Return [] if no violations found on this page."""


def _build_vision_prompt(checkpoints: list[dict], page_num: int, workflow_id: str = "edit") -> str:
    """Build the vision AI prompt for checking a single page image."""
    rules = "\n".join(
        f"{i + 1}. [{cp['id']}] {cp['instructions'].strip()}"
        for i, cp in enumerate(checkpoints)
    )
    if workflow_id in ("math", "mae_and_mathematica", "umq"):
        return MATH_PROMPT.format(rules=rules, page_num=page_num)
    elif workflow_id == "hse":
        return HSE_PROMPT.format(rules=rules, page_num=page_num)
    else:
        return EDIT_PROMPT.format(rules=rules, page_num=page_num)


def _extract_complete_objects(text: str) -> list[dict]:
    """
    Walk the text character-by-character and extract every complete JSON object.
    Works correctly even when the outer array is truncated mid-response.
    """
    findings = []
    depth = 0
    start = -1
    in_string = False
    escape_next = False

    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    obj = json.loads(text[start : i + 1])
                    if isinstance(obj, dict):
                        findings.append(obj)
                except json.JSONDecodeError:
                    pass
                start = -1
    return findings


def _parse_response(raw: str) -> list[dict]:
    raw = raw.strip()
    # Strip markdown code fences if the model included them
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    # 1. Try clean full parse
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
        return []
    except json.JSONDecodeError:
        pass

    # 2. Try to extract a complete JSON array (handles leading/trailing noise)
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # 3. Response was truncated — extract every complete object we can find
    return _extract_complete_objects(raw)


def _run_gemini_vision(image_bytes: bytes, checkpoints: list[dict], page_num: int, config: dict, workflow_id: str = "edit") -> list[dict]:
    """Call Gemini with vision capabilities."""
    import google.generativeai as genai

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"API key not found. Set the environment variable '{config['api_key_env']}'."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=config["model"],
        generation_config=genai.types.GenerationConfig(
            temperature=config["temperature"],
            max_output_tokens=config["max_tokens"],
        ),
    )

    # Convert bytes to PIL Image
    image = Image.open(BytesIO(image_bytes))

    prompt = _build_vision_prompt(checkpoints, page_num, workflow_id)
    response = model.generate_content([prompt, image])
    findings = _parse_response(response.text)
    return findings


def _run_anthropic_vision(image_bytes: bytes, checkpoints: list[dict], page_num: int, config: dict, workflow_id: str = "edit") -> list[dict]:
    """Call Anthropic Claude with vision capabilities."""
    import anthropic
    import base64

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"API key not found. Set the environment variable '{config['api_key_env']}'."
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Encode image as base64
    image_data = base64.standard_b64encode(image_bytes).decode("utf-8")

    prompt = _build_vision_prompt(checkpoints, page_num, workflow_id)
    message = client.messages.create(
        model=config["model"],
        max_tokens=config["max_tokens"],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )
    raw = message.content[0].text
    findings = _parse_response(raw)
    return findings


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


def _build_review_prompt(findings: list[dict]) -> str:
    lines = [
        f'{i + 1}. [finding_id: {f["id"]}] Quote: "{f["quote"]}" | Issue: {f["issue"]}'
        for i, f in enumerate(findings)
    ]
    return REVIEW_PROMPT.format(findings_list="\n".join(lines))


def _parse_review_response(raw: str) -> list[dict]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
        return []
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return _extract_complete_objects(raw)


def _run_gemini_review(image_bytes: bytes, prompt: str, config: dict) -> list[dict]:
    import google.generativeai as genai
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
    return _parse_review_response(response.text)


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
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return _parse_review_response(message.content[0].text)


def run_vision_review(image_bytes: bytes, findings: list[dict], page_num: int) -> list[dict]:
    """
    Second-pass review: validates each finding from the first pass.
    Returns list of {finding_id, verdict, reason} dicts.
    Skipped automatically if findings is empty.
    """
    if not findings:
        return []
    config = _load_model_config()
    provider = config.get("provider", "gemini")
    prompt = _build_review_prompt(findings)
    if provider == "gemini":
        reviews = _run_gemini_review(image_bytes, prompt, config)
    elif provider == "anthropic":
        reviews = _run_anthropic_review(image_bytes, prompt, config)
    else:
        return []
    required_keys = {"finding_id", "verdict", "reason"}
    return [r for r in reviews if isinstance(r, dict) and required_keys.issubset(r.keys())]


# ── Document-level check (full PDF as input) ──────────────────────────────────

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
    return _parse_response(response.text)


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
        messages=[{
            "role": "user",
            "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_data}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return _parse_response(message.content[0].text)


def run_document_check(pdf_bytes: bytes, checkpoints: list[dict]) -> list[dict]:
    """
    Document-level check: sends the full PDF to the AI with document-scope checkpoints.
    Returns findings list same schema as run_vision_check.
    """
    if not checkpoints:
        return []
    config = _load_model_config()
    provider = config.get("provider", "gemini")
    prompt = _build_document_prompt(checkpoints)
    if provider == "gemini":
        findings = _run_gemini_document(pdf_bytes, prompt, config)
    elif provider == "anthropic":
        findings = _run_anthropic_document(pdf_bytes, prompt, config)
    else:
        return []
    required_keys = {"checkpoint_id", "quote", "location", "issue", "suggestion"}
    return [f for f in findings if isinstance(f, dict) and required_keys.issubset(f.keys())]


def run_document_review(pdf_bytes: bytes, findings: list[dict]) -> list[dict]:
    """
    Second-pass review for document-level findings using the same PDF.
    Skipped automatically if findings is empty.
    """
    if not findings:
        return []
    config = _load_model_config()
    provider = config.get("provider", "gemini")
    prompt = _build_document_review_prompt(findings)
    if provider == "gemini":
        reviews = _run_gemini_document(pdf_bytes, prompt, config)
    elif provider == "anthropic":
        reviews = _run_anthropic_document(pdf_bytes, prompt, config)
    else:
        return []
    required_keys = {"finding_id", "verdict", "reason"}
    return [r for r in reviews if isinstance(r, dict) and required_keys.issubset(r.keys())]


def run_vision_check(image_bytes: bytes, checkpoints: list[dict], page_num: int, workflow_id: str = "edit") -> list[dict]:
    """
    Runs vision-based checking on a single page image.

    Args:
        image_bytes: JPEG image bytes
        checkpoints: list of checkpoint dicts with 'id', 'name', 'description'
        page_num: page number (for location field in findings)
        workflow_id: workflow identifier (e.g., 'edit', 'math')

    Returns:
        list of finding dicts, each with:
            checkpoint_id, quote, location, issue, suggestion
    """
    config = _load_model_config()
    provider = config.get("provider", "gemini")

    if provider == "gemini":
        findings = _run_gemini_vision(image_bytes, checkpoints, page_num, config, workflow_id)
    elif provider == "anthropic":
        findings = _run_anthropic_vision(image_bytes, checkpoints, page_num, config, workflow_id)
    else:
        raise ValueError(
            f"Unsupported provider '{provider}' in config/model_config.yaml. "
            "Supported values: gemini, anthropic"
        )

    # Validate that each finding has the required fields; drop malformed ones
    required_keys = {"checkpoint_id", "quote", "location", "issue", "suggestion"}
    valid = [
        f for f in findings
        if isinstance(f, dict) and required_keys.issubset(f.keys())
    ]
    return valid
