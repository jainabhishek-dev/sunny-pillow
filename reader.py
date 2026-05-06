import re
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# MIME types the app supports and what reader to use for each
SUPPORTED_TYPES = {
    "application/vnd.google-apps.document": "google_doc",
    "application/vnd.google-apps.presentation": "google_slides",
    "application/pdf": "pdf",
}

# MIME types that are explicitly not supported — give user a clear message
UNSUPPORTED_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        "Word (.docx) files are not supported directly. "
        "Please open the file in Google Drive, then go to File → Save as Google Docs, and share that link instead."
    ),
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": (
        "PowerPoint (.pptx) files are not supported directly. "
        "Please open the file in Google Drive, then go to File → Save as Google Slides, and share that link instead."
    ),
}


def extract_file_id(url: str) -> str:
    patterns = [
        r"/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(
        "Could not extract a file ID from the URL you provided. "
        "Please paste the full Google Drive or Docs/Slides share link."
    )


def _build_credentials(token: dict) -> Credentials:
    return Credentials(token=token["access_token"])


def _build_drive_service(creds: Credentials):
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _build_docs_service(creds: Credentials):
    return build("docs", "v1", credentials=creds, cache_discovery=False)


def _build_slides_service(creds: Credentials):
    return build("slides", "v1", credentials=creds, cache_discovery=False)


def _read_google_doc(docs_service, file_id: str) -> tuple[list[dict], str, list[tuple]]:
    doc = docs_service.documents().get(documentId=file_id).execute()
    sections = []
    text_runs = []  # (doc_start_index, run_text) — used for comment anchoring
    paragraph_index = 0

    for element in doc.get("body", {}).get("content", []):
        if "paragraph" in element:
            paragraph_index += 1
            text_parts = []
            for part in element["paragraph"].get("elements", []):
                run = part.get("textRun", {})
                content = run.get("content", "")
                start_idx = part.get("startIndex", 0)
                if content:
                    # Always include the run in the paragraph text — even a
                    # whitespace-only run matters when it separates two words
                    # that have different formatting (e.g. bold/colour boundary).
                    # Dropping it would merge the adjacent words and cause false
                    # "missing space" findings from the AI.
                    text_parts.append(content)
                if content.strip():
                    # Only index non-whitespace runs for comment anchoring.
                    text_runs.append((start_idx, content))
            text = "".join(text_parts).strip()
            if text:
                sections.append({"location": f"Paragraph {paragraph_index}", "text": text})

    full_text = "\n\n".join(
        f"[{s['location']}] {s['text']}" for s in sections
    )
    return sections, full_text, text_runs


def _read_google_slides(slides_service, file_id: str) -> tuple[list[dict], str]:
    presentation = slides_service.presentations().get(presentationId=file_id).execute()
    sections = []

    for slide_index, slide in enumerate(presentation.get("slides", []), start=1):
        slide_texts = []
        for element in slide.get("pageElements", []):
            shape = element.get("shape", {})
            text_content = shape.get("text", {})
            for text_element in text_content.get("textElements", []):
                run = text_element.get("textRun", {})
                content = run.get("content", "")
                if content:
                    # Preserve the run as-is (including whitespace-only runs)
                    # so formatting-boundary spaces are not lost.
                    slide_texts.append(content)

        if slide_texts:
            # Concatenate runs directly (each run already contains its own
            # spacing / newlines); strip trailing whitespace for cleanliness.
            slide_text = "".join(slide_texts).strip()
            if slide_text:
                sections.append({
                    "location": f"Slide {slide_index}",
                    "text": slide_text,
                })

    full_text = "\n\n".join(
        f"[{s['location']}] {s['text']}" for s in sections
    )
    return sections, full_text


def _read_pdf(drive_service, file_id: str) -> tuple[list[dict], str]:
    content = (
        drive_service.files()
        .export(fileId=file_id, mimeType="text/plain", supportsAllDrives=True)
        .execute()
    )
    raw_text = content.decode("utf-8") if isinstance(content, bytes) else content
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    # Group into paragraphs of ~5 lines each for location reporting
    sections = []
    chunk_size = 5
    for i in range(0, len(lines), chunk_size):
        chunk = " ".join(lines[i : i + chunk_size])
        paragraph_num = (i // chunk_size) + 1
        sections.append({"location": f"Paragraph {paragraph_num}", "text": chunk})

    full_text = "\n\n".join(f"[{s['location']}] {s['text']}" for s in sections)
    return sections, full_text


def get_file_content(token: dict, drive_url: str) -> dict:
    """
    Fetches and returns structured content from a Google Drive file.

    Returns a dict with:
        type       - 'google_doc' | 'google_slides' | 'pdf'
        file_id    - the Drive file ID
        title      - the document title
        sections   - list of {location, text} dicts
        full_text  - flat string passed to the AI
    """
    file_id = extract_file_id(drive_url)
    creds = _build_credentials(token)
    drive_service = _build_drive_service(creds)

    metadata = (
        drive_service.files()
        .get(fileId=file_id, fields="id,name,mimeType", supportsAllDrives=True)
        .execute()
    )
    mime_type = metadata.get("mimeType", "")
    title = metadata.get("name", "Untitled")

    if mime_type in UNSUPPORTED_TYPES:
        raise ValueError(UNSUPPORTED_TYPES[mime_type])

    if mime_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"This file type ({mime_type}) is not supported. "
            "Please provide a link to a Google Doc, Google Slides presentation, or PDF."
        )

    file_type = SUPPORTED_TYPES[mime_type]

    text_runs: list[tuple] = []

    if file_type == "google_doc":
        docs_service = _build_docs_service(creds)
        sections, full_text, text_runs = _read_google_doc(docs_service, file_id)
    elif file_type == "google_slides":
        slides_service = _build_slides_service(creds)
        sections, full_text = _read_google_slides(slides_service, file_id)
    elif file_type == "pdf":
        sections, full_text = _read_pdf(drive_service, file_id)

    if not full_text.strip():
        raise ValueError(
            f"No readable text was found in '{title}'. "
            "The file may be empty or contain only images."
        )

    return {
        "type": file_type,
        "file_id": file_id,
        "title": title,
        "sections": sections,
        "full_text": full_text,
        "text_runs": text_runs,
    }
