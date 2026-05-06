import json

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


def _build_drive_service(token: dict):
    creds = Credentials(token=token["access_token"])
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _format_comment(finding: dict, checkpoint_map: dict) -> str:
    cp = checkpoint_map.get(finding["checkpoint_id"], {})
    cp_name = cp.get("name", finding["checkpoint_id"])
    lines = [
        f"[{finding['checkpoint_id']}] {cp_name}",
        f"",
        f"Location: {finding['location']}",
        f"Quoted text: \"{finding['quote']}\"",
        f"",
        f"Issue: {finding['issue']}",
        f"Suggestion: {finding['suggestion']}",
    ]
    return "\n".join(lines)


def _find_doc_offset(text_runs: list[tuple], quote: str) -> int | None:
    """
    Find the document character offset (startIndex) of a quote by scanning
    all recorded text runs. Returns None if the quote isn't found in any run.
    """
    for start_idx, run_text in text_runs:
        pos = run_text.find(quote)
        if pos != -1:
            return start_idx + pos
    return None


def _add_comment_to_doc(
    drive_service,
    file_id: str,
    content: str,
    quoted_text: str,
    text_runs: list[tuple] | None = None,
) -> bool:
    """
    Adds an anchored comment to a Google Doc.
    Uses real document character offsets from text_runs for accurate anchoring.
    Falls back to an unanchored comment if anchoring fails.
    Returns True on success.
    """
    snippet = quoted_text[:100]
    offset = _find_doc_offset(text_runs, snippet) if text_runs else None
    anchor = json.dumps({
        "r": "head",
        "a": [{"ct": snippet, "s": offset if offset is not None else 0}],
    })

    try:
        drive_service.comments().create(
            fileId=file_id,
            body={"content": content, "anchor": anchor},
            fields="id",
        ).execute()
        return True
    except Exception:
        # Anchor failed (text not found verbatim, special chars, etc.) — post without anchor
        try:
            drive_service.comments().create(
                fileId=file_id,
                body={"content": content},
                fields="id",
            ).execute()
            return True
        except Exception:
            return False


def _add_comment_to_slides(drive_service, file_id: str, content: str) -> bool:
    """
    Adds a file-level comment to a Google Slides presentation.
    Drive API does not support anchoring comments to individual slides.
    Returns True on success.
    """
    try:
        drive_service.comments().create(
            fileId=file_id,
            body={"content": content},
            fields="id",
        ).execute()
        return True
    except Exception:
        return False


def _pick_anchor_text(file_data: dict) -> str:
    """
    Picks a 20–60 character anchor from the first usable section of the document.
    Falls back to 'the' if nothing suitable is found.
    """
    for section in file_data.get("sections", []):
        text = section.get("text", "").strip()
        if len(text) >= 20:
            # Take a clean slice of 20–60 chars, not cutting mid-word
            snippet = text[:60]
            last_space = snippet.rfind(" ", 20)
            return snippet[:last_space] if last_space != -1 else snippet
    return "the"


def post_test_comment(token: dict, file_data: dict) -> dict:
    """
    Posts a single test comment to verify Drive API access and comment anchoring.
    Anchors to the first real sentence fragment (20–60 chars) from the document.
    For Google Slides: posts a file-level comment.
    Returns {"success": True, "anchor": "..."} or {"success": False, "error": "..."}.
    """
    if file_data["type"] == "pdf":
        return {"success": False, "error": "PDF files do not support Drive comments."}

    drive_service = _build_drive_service(token)
    file_id = file_data["file_id"]
    anchor_text = _pick_anchor_text(file_data)
    content = f"Test comment from Sunny Pillow — API access confirmed. You can delete this."

    try:
        if file_data["type"] == "google_doc":
            success = _add_comment_to_doc(drive_service, file_id, content, anchor_text, file_data.get("text_runs"))
        elif file_data["type"] == "google_slides":
            success = _add_comment_to_slides(drive_service, file_id, content)
        else:
            success = False
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    if success:
        return {"success": True, "anchor": anchor_text}
    return {
        "success": False,
        "error": "Comment could not be posted. Check that you have edit access to the file.",
    }


def post_comments(token: dict, file_data: dict, findings: list[dict], checkpoint_map: dict) -> int:
    """
    Posts each finding as a Drive comment on the file.
    For PDFs, no comments are posted (Drive API does not support it).
    Returns the number of comments successfully posted.
    """
    if file_data["type"] == "pdf":
        return 0

    drive_service = _build_drive_service(token)
    file_id = file_data["file_id"]
    file_type = file_data["type"]
    posted = 0

    for finding in findings:
        comment_text = _format_comment(finding, checkpoint_map)

        if file_type == "google_doc":
            success = _add_comment_to_doc(
                drive_service, file_id, comment_text, finding["quote"],
                file_data.get("text_runs"),
            )
        elif file_type == "google_slides":
            success = _add_comment_to_slides(drive_service, file_id, comment_text)
        else:
            success = False

        if success:
            posted += 1

    return posted
