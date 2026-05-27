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



# ── Vision prompt template (fully hardcoded) ─────────────────────────────────
#
# {workflow_name}, {rules}, and {page_num} are replaced at runtime via
# str.replace() — no .format() to avoid brace-escaping issues.

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

# ── Structured-output schema for workflow generation ──────────────────────────
#
# Only checkpoints are generated — the reviewer persona is constructed from
# the workflow name at runtime, so there is nothing for the AI to generate there.

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


def _build_vision_prompt(checkpoints: list[dict], page_num: int, workflow_name: str) -> str:
    """Build the full vision prompt from the hardcoded template and runtime values."""
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


def _run_gemini_vision(image_bytes: bytes, checkpoints: list[dict], page_num: int, config: dict, workflow_name: str) -> list[dict]:
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

    prompt = _build_vision_prompt(checkpoints, page_num, workflow_name)
    response = model.generate_content([prompt, image])
    findings = _parse_response(response.text)
    return findings


def _run_anthropic_vision(image_bytes: bytes, checkpoints: list[dict], page_num: int, config: dict, workflow_name: str) -> list[dict]:
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

    prompt = _build_vision_prompt(checkpoints, page_num, workflow_name)
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


def run_vision_check(image_bytes: bytes, checkpoints: list[dict], page_num: int, workflow_name: str) -> list[dict]:
    """
    Runs vision-based checking on a single page image.

    Args:
        image_bytes: JPEG image bytes
        checkpoints: list of checkpoint dicts with 'id', 'instructions', etc.
        page_num: page number (for location field in findings)
        workflow_name: display name of the workflow (e.g. "HSE"), used to
                       construct the reviewer persona in the prompt

    Returns:
        list of finding dicts, each with:
            checkpoint_id, quote, location, issue, suggestion
    """
    config = _load_model_config()
    provider = config.get("provider", "gemini")

    if provider == "gemini":
        findings = _run_gemini_vision(image_bytes, checkpoints, page_num, config, workflow_name)
    elif provider == "anthropic":
        findings = _run_anthropic_vision(image_bytes, checkpoints, page_num, config, workflow_name)
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


# ── Workflow generation ───────────────────────────────────────────────────────

# ── CIC (Comment Incorporation Check) ────────────────────────────────────────

_CIC_PAGE_PROMPT_TEMPLATE = """\
You are checking whether reviewer comments from an original document have been addressed in a revised version.

LEFT IMAGE: Page {page_num} of the ORIGINAL document (the one reviewers commented on).
RIGHT IMAGE: Page {page_num} of the REVISED document (should have addressed the comments).

Reviewer comments to check on this page:
{comments_list}

For each comment, determine whether it has been incorporated in the revised version.

Verdicts:
- "fixed": The comment has been clearly addressed in the revised document
- "not_fixed": The revised document still has the same issue the comment pointed out
- "not_sure": Cannot determine from this page alone whether the comment was addressed

Return a JSON array only. No markdown, no explanation.
Schema:
[{{"comment_id": "...", "verdict": "fixed", "reason": "..."}}]

Keep reason to 15 words or fewer. Include every comment_id in your response.\
"""

_CIC_GLOBAL_PROMPT_TEMPLATE = """\
You are checking whether reviewer comments from an original document have been addressed in a revised version.

DOCUMENT 1 (first attachment): The ORIGINAL document with reviewer comments.
DOCUMENT 2 (second attachment): The REVISED document that should have addressed all comments.

The following comments could not be conclusively resolved from individual pages and need full-document context:
{comments_list}

For each comment, determine whether it has been incorporated in the revised version by examining both full documents.

Verdicts:
- "fixed": The comment has been clearly addressed in the revised document
- "not_fixed": The revised document still has the same issue the comment pointed out
- "not_sure": Cannot determine even from the full document context

Return a JSON array only. No markdown, no explanation.
Schema:
[{{"comment_id": "...", "verdict": "fixed", "reason": "..."}}]

Keep reason to 15 words or fewer. Include every comment_id in your response.\
"""


def _build_cic_comments_text(comments: list[dict]) -> str:
    lines = []
    for i, c in enumerate(comments, 1):
        author = c.get("author", "")
        content = c.get("content", "")
        cid = c.get("id", "")
        author_str = f" [{author}]" if author else ""
        lines.append(f'{i}. [comment_id: {cid}]{author_str} "{content}"')
    return "\n".join(lines)


def _parse_cic_response(raw: str) -> list[dict]:
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


def _run_gemini_cic_page(f1_img_bytes: bytes, f2_img_bytes: bytes, page_num: int, comments: list[dict], config: dict) -> list[dict]:
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
    f1_image = Image.open(BytesIO(f1_img_bytes))
    f2_image = Image.open(BytesIO(f2_img_bytes))
    comments_text = _build_cic_comments_text(comments)
    prompt = (
        _CIC_PAGE_PROMPT_TEMPLATE
        .replace("{page_num}", str(page_num))
        .replace("{comments_list}", comments_text)
    )
    response = model.generate_content([prompt, f1_image, f2_image])
    return _parse_cic_response(response.text)


def _run_anthropic_cic_page(f1_img_bytes: bytes, f2_img_bytes: bytes, page_num: int, comments: list[dict], config: dict) -> list[dict]:
    import anthropic
    import base64
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"API key not found. Set '{config['api_key_env']}'.")
    client = anthropic.Anthropic(api_key=api_key)
    f1_data = base64.standard_b64encode(f1_img_bytes).decode("utf-8")
    f2_data = base64.standard_b64encode(f2_img_bytes).decode("utf-8")
    comments_text = _build_cic_comments_text(comments)
    prompt = (
        _CIC_PAGE_PROMPT_TEMPLATE
        .replace("{page_num}", str(page_num))
        .replace("{comments_list}", comments_text)
    )
    message = client.messages.create(
        model=config["model"],
        max_tokens=config["max_tokens"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "LEFT IMAGE (original document):"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": f1_data}},
                {"type": "text", "text": "RIGHT IMAGE (revised document):"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": f2_data}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return _parse_cic_response(message.content[0].text)


def run_cic_check(f1_img_bytes: bytes, f2_img_bytes: bytes, page_num: int, comments: list[dict]) -> list[dict]:
    """
    Check whether reviewer comments anchored to page N of the original doc are
    addressed in page N of the revised doc.

    Args:
        f1_img_bytes: JPEG bytes for page N of the original (commented) document
        f2_img_bytes: JPEG bytes for page N of the revised document
        page_num: page number (used in prompt context)
        comments: list of {id, content, author} dicts for comments anchored to this page

    Returns:
        list of {comment_id, verdict, reason} dicts
        verdict is "fixed" | "not_fixed" | "not_sure"
    """
    if not comments:
        return []
    config = _load_model_config()
    provider = config.get("provider", "gemini")
    if provider == "gemini":
        results = _run_gemini_cic_page(f1_img_bytes, f2_img_bytes, page_num, comments, config)
    elif provider == "anthropic":
        results = _run_anthropic_cic_page(f1_img_bytes, f2_img_bytes, page_num, comments, config)
    else:
        return []
    required_keys = {"comment_id", "verdict", "reason"}
    return [r for r in results if isinstance(r, dict) and required_keys.issubset(r.keys())]


def _run_gemini_cic_global(pdf1_bytes: bytes, pdf2_bytes: bytes, comments: list[dict], config: dict) -> list[dict]:
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
    pdf1_data = base64.standard_b64encode(pdf1_bytes).decode("utf-8")
    pdf2_data = base64.standard_b64encode(pdf2_bytes).decode("utf-8")
    comments_text = _build_cic_comments_text(comments)
    prompt = _CIC_GLOBAL_PROMPT_TEMPLATE.replace("{comments_list}", comments_text)
    response = model.generate_content([
        {"mime_type": "application/pdf", "data": pdf1_data},
        {"mime_type": "application/pdf", "data": pdf2_data},
        prompt,
    ])
    return _parse_cic_response(response.text)


def _run_anthropic_cic_global(pdf1_bytes: bytes, pdf2_bytes: bytes, comments: list[dict], config: dict) -> list[dict]:
    import anthropic
    import base64
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"API key not found. Set '{config['api_key_env']}'.")
    client = anthropic.Anthropic(api_key=api_key)
    pdf1_data = base64.standard_b64encode(pdf1_bytes).decode("utf-8")
    pdf2_data = base64.standard_b64encode(pdf2_bytes).decode("utf-8")
    comments_text = _build_cic_comments_text(comments)
    prompt = _CIC_GLOBAL_PROMPT_TEMPLATE.replace("{comments_list}", comments_text)
    message = client.messages.create(
        model=config["model"],
        max_tokens=config["max_tokens"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf1_data}},
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf2_data}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return _parse_cic_response(message.content[0].text)


def run_cic_global_check(pdf1_bytes: bytes, pdf2_bytes: bytes, not_sure_comments: list[dict]) -> list[dict]:
    """
    Final full-document pass for comments still "not_sure" after page-by-page processing.
    Sends both complete PDFs to the AI together with all remaining unresolved comments.

    Args:
        pdf1_bytes: full PDF bytes of the original (commented) document
        pdf2_bytes: full PDF bytes of the revised document
        not_sure_comments: list of {id, content, author} dicts still unresolved

    Returns:
        list of {comment_id, verdict, reason} dicts
    """
    if not not_sure_comments:
        return []
    config = _load_model_config()
    provider = config.get("provider", "gemini")
    if provider == "gemini":
        results = _run_gemini_cic_global(pdf1_bytes, pdf2_bytes, not_sure_comments, config)
    elif provider == "anthropic":
        results = _run_anthropic_cic_global(pdf1_bytes, pdf2_bytes, not_sure_comments, config)
    else:
        return []
    required_keys = {"comment_id", "verdict", "reason"}
    return [r for r in results if isinstance(r, dict) and required_keys.issubset(r.keys())]


# ── Workflow generation ───────────────────────────────────────────────────────

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

    config = _load_model_config()

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
