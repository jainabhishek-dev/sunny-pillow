"""AI functions for the CIC (Comment Incorporation Check) workflow."""

import os
from io import BytesIO

from services.ai_client import load_model_config, parse_response

# ── Prompts ────────────────────────────────────────────────────────────────────

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


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_cic_comments_text(comments: list[dict]) -> str:
    lines = []
    for i, c in enumerate(comments, 1):
        author = c.get("author", "")
        content = c.get("content", "")
        cid = c.get("id", "")
        author_str = f" [{author}]" if author else ""
        lines.append(f'{i}. [comment_id: {cid}]{author_str} "{content}"')
    return "\n".join(lines)


# ── Gemini runners ─────────────────────────────────────────────────────────────

def _run_gemini_cic_page(f1_img_bytes: bytes, f2_img_bytes: bytes, page_num: int, comments: list[dict], config: dict) -> list[dict]:
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
    f1_image = Image.open(BytesIO(f1_img_bytes))
    f2_image = Image.open(BytesIO(f2_img_bytes))
    comments_text = _build_cic_comments_text(comments)
    prompt = (
        _CIC_PAGE_PROMPT_TEMPLATE
        .replace("{page_num}", str(page_num))
        .replace("{comments_list}", comments_text)
    )
    response = model.generate_content([prompt, f1_image, f2_image])
    return parse_response(response.text)


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
    return parse_response(response.text)


# ── Anthropic runners ──────────────────────────────────────────────────────────

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
    return parse_response(message.content[0].text)


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
    return parse_response(message.content[0].text)


# ── Public API ─────────────────────────────────────────────────────────────────

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
    config = load_model_config()
    provider = config.get("provider", "gemini")
    if provider == "gemini":
        results = _run_gemini_cic_page(f1_img_bytes, f2_img_bytes, page_num, comments, config)
    elif provider == "anthropic":
        results = _run_anthropic_cic_page(f1_img_bytes, f2_img_bytes, page_num, comments, config)
    else:
        return []
    required_keys = {"comment_id", "verdict", "reason"}
    return [r for r in results if isinstance(r, dict) and required_keys.issubset(r.keys())]


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
    config = load_model_config()
    provider = config.get("provider", "gemini")
    if provider == "gemini":
        results = _run_gemini_cic_global(pdf1_bytes, pdf2_bytes, not_sure_comments, config)
    elif provider == "anthropic":
        results = _run_anthropic_cic_global(pdf1_bytes, pdf2_bytes, not_sure_comments, config)
    else:
        return []
    required_keys = {"comment_id", "verdict", "reason"}
    return [r for r in results if isinstance(r, dict) and required_keys.issubset(r.keys())]
