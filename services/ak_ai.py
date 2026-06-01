"""AI functions for the AK (Answer Key) Review workflow.

Two-phase approach:
  Phase 1 — extract_ak_questions: chapter PDF only → list of question stubs
  Phase 2 — review_ak_exercise:   both PDFs + prompt → per-question verdicts for one exercise
"""

import json
import os
import base64

from services.ai_client import load_model_config, parse_response


# ── Default prompt ─────────────────────────────────────────────────────────────

AK_REVIEW_DEFAULT_PROMPT = """\
You are an expert Mathematics Educator and Answer Key Reviewer.

Your task is to review the Answer Key against the Chapter PDF and identify any missing or incorrect answers.

Column Rules:

Present in AK?
Use: Yes / No

Answer Correct?
Use: Yes / No / Manual Review Required

Use "Manual Review Required" only when:
- The answer depends on a diagram, graph, table, construction, or image that cannot be interpreted reliably.
- The question is unclear due to scan quality or missing information.
- The question requires visual verification that cannot be performed confidently.

Suggestions:
- If Present in AK = No: Mention "Answer missing in AK." Provide the correct answer wherever possible.
- If Answer Correct? = No: Provide the correct answer.
- If Answer Correct? = Manual Review Required: Clearly explain the issue requiring manual verification.
Keep suggestions concise and correction-focused.

Mathematics Accuracy Rules:

Numerical Answers:
- Recalculate the answer independently.
- Check all arithmetic operations carefully.
- Verify signs (+, -), decimal placement, and place value.
- Verify rounding and estimation requirements wherever applicable.
- Check units and unit conversions.
- Ensure the final answer matches the form requested in the question.

Fractions and Decimals:
- Verify calculations independently.
- Ensure fractions are simplified when required.
- Accept equivalent fractions unless the question explicitly requires simplest form.
- Verify conversion between fractions, decimals, and percentages.

Algebra:
- Substitute values back into the original equation whenever possible.
- Check that all solutions satisfy the given conditions.
- Verify expansion, factorisation, simplification, and algebraic manipulation.
- Check for extraneous solutions.

Geometry and Mensuration:
- Verify formulas used.
- Recalculate measurements independently.
- Check units carefully.
- Verify that the answer corresponds to the quantity asked.
- Verify geometric properties, angle measures, and constructions wherever possible.

Data Handling and Statistics:
- Recalculate totals, averages, percentages, ratios, mean, median, and mode wherever applicable.
- Verify interpretation of tables, graphs, pictographs, and charts.
- Ensure scales and labels have been interpreted correctly.

Graphs, Tables, and Diagrams:
- Do not mark an answer correct solely because it appears visually similar.
- Verify values, labels, scales, coordinates, plotted points, and data representation.
- If the graphical answer cannot be reliably verified, mark as "Manual Review Required".

Word Problems:
- Independently solve the complete problem.
- Verify that the final answer addresses the actual question asked.
- Check units and context.
- Ensure intermediate calculation errors have not resulted in an incorrect final answer.

Multiple-Part Questions:
- Verify every subpart independently.
- Mark the answer incorrect if any required subpart is missing or incorrect.
- Clearly identify the incorrect subpart.

Matching, Fill in the Blanks, True/False, and MCQs:
- Independently verify each item.
- Do not assume the provided answer is correct.
- Check all options before validating an MCQ answer.

Answer Format Validation:
Treat answer-format errors as incorrect when the question explicitly requires a specific format, such as:
- Fraction instead of decimal
- Simplest form
- Ascending order / Descending order
- Units
- Labelled diagram or graph
- Table format
- Specific notation requested in the question

Final Validation Rule:
Never mark an answer as correct unless it has been independently verified mathematically.
If there is uncertainty, use "Manual Review Required" rather than guessing.

Important Instructions:
- Do not skip any exercise question.
- Do not stop after finding an error.
- Solve questions independently before comparing with the Answer Key.
- If a question is missing from the Answer Key, still solve it and provide the correct answer.
- Be especially careful with signs, units, decimal values, fractions, place value, ordering, graphs, tables, and geometry-related answers.
- Treat every subpart as a separate answer-checking unit.\
"""


# ── Prompts ────────────────────────────────────────────────────────────────────

_QUESTION_EXTRACTION_PROMPT = """\
You are reading a Mathematics chapter PDF. Extract ONLY questions from Exercises and Additional Exercises.

DO NOT extract questions from:
- Solved Examples / Worked Examples
- Activities / Projects / Investigations / Lab Activities
- Side Boxes / Fun Facts / Think and Discuss
- Warm-up Sections / Concept Checks
- Any other non-exercise content

Treat every subpart independently. Examples: 1, 2(a), 2(b), 3(c)(i)

Return a JSON array only. No markdown, no explanation.
Each element must have:
  "page_no": integer (page number in the chapter where the question appears),
  "exercise_no": string (e.g. "Exercise 5A", "Exercise 5B", "Additional Exercise A"),
  "question_no": string (e.g. "1", "2(a)", "3(b)(i)")

Example:
[
  {"page_no": 3, "exercise_no": "Exercise 5A", "question_no": "1"},
  {"page_no": 3, "exercise_no": "Exercise 5A", "question_no": "2(a)"},
  {"page_no": 5, "exercise_no": "Exercise 5B", "question_no": "1"}
]\
"""

_EXERCISE_REVIEW_PROMPT_TEMPLATE = """\
DOCUMENT 1 (first attachment): The CHAPTER PDF (contains the questions).
DOCUMENT 2 (second attachment): The ANSWER KEY PDF (contains the answers to check).

Exercise to review: {exercise_no}

Questions from this exercise that need to be reviewed (already extracted from the chapter):
{questions_json}

For each question above:
1. Locate the question in DOCUMENT 1 (Chapter PDF).
2. Locate the corresponding answer in DOCUMENT 2 (Answer Key PDF).
3. Independently solve the question.
4. Fill in: present_in_ak, answer_correct, suggestions.

--- Review Instructions ---
{review_prompt}
--- End of Review Instructions ---

Return a JSON array only. No markdown, no explanation.
Return one object per question in the same order as the input list.
Schema:
[{{"page_no": <int>, "exercise_no": "<str>", "question_no": "<str>", "present_in_ak": "Yes|No", "answer_correct": "Yes|No|Manual Review Required", "suggestions": "<str or null>"}}]\
"""


# ── Gemini runners ─────────────────────────────────────────────────────────────

def _run_gemini_extract(chapter_bytes: bytes, config: dict) -> list[dict]:
    import google.generativeai as genai

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"API key not found. Set '{config['api_key_env']}'.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=config["model"],
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=config["max_tokens"],
        ),
    )
    chapter_data = base64.standard_b64encode(chapter_bytes).decode("utf-8")
    response = model.generate_content([
        {"mime_type": "application/pdf", "data": chapter_data},
        _QUESTION_EXTRACTION_PROMPT,
    ])
    return parse_response(response.text)


def _run_gemini_review_exercise(
    chapter_bytes: bytes,
    ak_bytes: bytes,
    exercise_no: str,
    questions: list[dict],
    review_prompt: str,
    config: dict,
) -> list[dict]:
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
    chapter_data = base64.standard_b64encode(chapter_bytes).decode("utf-8")
    ak_data = base64.standard_b64encode(ak_bytes).decode("utf-8")
    prompt = (
        _EXERCISE_REVIEW_PROMPT_TEMPLATE
        .replace("{exercise_no}", exercise_no)
        .replace("{questions_json}", json.dumps(questions, indent=2))
        .replace("{review_prompt}", review_prompt)
    )
    response = model.generate_content([
        {"mime_type": "application/pdf", "data": chapter_data},
        {"mime_type": "application/pdf", "data": ak_data},
        prompt,
    ])
    return parse_response(response.text)


# ── Anthropic runners ──────────────────────────────────────────────────────────

def _run_anthropic_extract(chapter_bytes: bytes, config: dict) -> list[dict]:
    import anthropic

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"API key not found. Set '{config['api_key_env']}'.")
    client = anthropic.Anthropic(api_key=api_key)
    chapter_data = base64.standard_b64encode(chapter_bytes).decode("utf-8")
    message = client.messages.create(
        model=config["model"],
        max_tokens=config["max_tokens"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": chapter_data}},
                {"type": "text", "text": _QUESTION_EXTRACTION_PROMPT},
            ],
        }],
    )
    return parse_response(message.content[0].text)


def _run_anthropic_review_exercise(
    chapter_bytes: bytes,
    ak_bytes: bytes,
    exercise_no: str,
    questions: list[dict],
    review_prompt: str,
    config: dict,
) -> list[dict]:
    import anthropic

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"API key not found. Set '{config['api_key_env']}'.")
    client = anthropic.Anthropic(api_key=api_key)
    chapter_data = base64.standard_b64encode(chapter_bytes).decode("utf-8")
    ak_data = base64.standard_b64encode(ak_bytes).decode("utf-8")
    prompt = (
        _EXERCISE_REVIEW_PROMPT_TEMPLATE
        .replace("{exercise_no}", exercise_no)
        .replace("{questions_json}", json.dumps(questions, indent=2))
        .replace("{review_prompt}", review_prompt)
    )
    message = client.messages.create(
        model=config["model"],
        max_tokens=config["max_tokens"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": chapter_data}},
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": ak_data}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return parse_response(message.content[0].text)


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_ak_questions(chapter_bytes: bytes) -> list[dict]:
    """
    Phase 1: Extract all exercise questions from the chapter PDF.

    Returns a list of question stubs:
        [{"page_no": int, "exercise_no": str, "question_no": str}, ...]
    """
    config = load_model_config()
    provider = config.get("provider", "gemini")
    if provider == "gemini":
        results = _run_gemini_extract(chapter_bytes, config)
    elif provider == "anthropic":
        results = _run_anthropic_extract(chapter_bytes, config)
    else:
        return []
    required = {"exercise_no", "question_no"}
    return [r for r in results if isinstance(r, dict) and required.issubset(r.keys())]


def review_ak_exercise(
    chapter_bytes: bytes,
    ak_bytes: bytes,
    exercise_no: str,
    questions: list[dict],
    review_prompt: str,
) -> list[dict]:
    """
    Phase 2: Review one exercise — compare questions against the AK PDF.

    Returns completed question rows:
        [{"page_no", "exercise_no", "question_no", "present_in_ak", "answer_correct", "suggestions"}, ...]
    """
    if not questions:
        return []
    config = load_model_config()
    provider = config.get("provider", "gemini")
    if provider == "gemini":
        results = _run_gemini_review_exercise(chapter_bytes, ak_bytes, exercise_no, questions, review_prompt, config)
    elif provider == "anthropic":
        results = _run_anthropic_review_exercise(chapter_bytes, ak_bytes, exercise_no, questions, review_prompt, config)
    else:
        return []
    required = {"question_no", "present_in_ak", "answer_correct"}
    return [r for r in results if isinstance(r, dict) and required.issubset(r.keys())]
