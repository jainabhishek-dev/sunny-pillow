import json
import os
import re
from pathlib import Path

import yaml


def _load_model_config() -> dict:
    config_path = Path(__file__).parent / "config" / "model_config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_prompt(document_text: str, batch: list[dict]) -> str:
    rules = "\n".join(
        f"{i + 1}. [{cp['id']}] {cp['name']}: {cp['description'].strip()}"
        for i, cp in enumerate(batch)
    )
    return f"""You are a professional editorial checker for LEAD, an educational publishing house.

Your task is to read the document below and check it against the specific style rules listed.

DOCUMENT:
{document_text}

STYLE RULES TO CHECK (check ONLY these — ignore all others):
{rules}

INSTRUCTIONS:
- Read the document carefully and thoroughly.
- For each rule, identify EVERY instance where the document violates it.
- Be strict. If something looks like it could be a violation, flag it — do not give the benefit of the doubt. It is better to flag a possible issue than to miss a real one.
- Do not be conservative. Editors rely on you to catch issues they might miss.
- Quote the exact text from the document where the violation occurs (20–80 characters).
- Provide the location as it appears in the document (e.g., "Paragraph 3", "Slide 2").
- Keep the "issue" field to 15 words or fewer. Be direct: state what is wrong.
- Keep the "suggestion" field to 15 words or fewer. State the exact fix.
- Do not explain, do not repeat the rule, do not add context — just the error and the fix.
- If a rule genuinely has no violations after careful reading, do not include it in the output.

Return a JSON array only. No explanation, no markdown fences, no text outside the array.

Schema:
[
  {{
    "checkpoint_id": "cp_001",
    "quote": "exact text from document where the violation is",
    "location": "Paragraph 3",
    "issue": "What is wrong (max 15 words)",
    "suggestion": "Exact fix to apply (max 15 words)"
  }}
]

If there are no violations for any of the checked rules, return an empty array: []"""


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


def _run_gemini(document_text: str, checkpoints: list[dict], config: dict) -> list[dict]:
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

    batch_size = config["batch_size"]
    all_findings = []

    for i in range(0, len(checkpoints), batch_size):
        batch = checkpoints[i : i + batch_size]
        prompt = _build_prompt(document_text, batch)
        response = model.generate_content(prompt)
        findings = _parse_response(response.text)
        all_findings.extend(findings)

    return all_findings


def _run_anthropic(document_text: str, checkpoints: list[dict], config: dict) -> list[dict]:
    import anthropic

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"API key not found. Set the environment variable '{config['api_key_env']}'."
        )

    client = anthropic.Anthropic(api_key=api_key)
    batch_size = config["batch_size"]
    all_findings = []

    for i in range(0, len(checkpoints), batch_size):
        batch = checkpoints[i : i + batch_size]
        prompt = _build_prompt(document_text, batch)
        message = client.messages.create(
            model=config["model"],
            max_tokens=config["max_tokens"],
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text
        findings = _parse_response(raw)
        all_findings.extend(findings)

    return all_findings


def run_checks(document_text: str, selected_checkpoints: list[dict]) -> list[dict]:
    """
    Runs all selected checkpoints against the document text in batches.
    Returns a list of finding dicts, each with:
        checkpoint_id, quote, location, issue, suggestion
    """
    config = _load_model_config()
    provider = config.get("provider", "gemini")

    if provider == "gemini":
        findings = _run_gemini(document_text, selected_checkpoints, config)
    elif provider == "anthropic":
        findings = _run_anthropic(document_text, selected_checkpoints, config)
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
