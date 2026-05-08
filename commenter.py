import json

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


def _build_drive_service(token: dict):
    creds = Credentials(token=token["access_token"])
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _format_comment(finding: dict, checkpoint_map: dict) -> str:
    """Format a finding as a Drive comment."""
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


def _add_comment_to_doc(
    drive_service,
    file_id: str,
    content: str,
) -> bool:
    """
    Adds an unanchored comment to a Google Doc.
    Returns True on success, False on failure.
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


def post_selected_comments(token: dict, file_data: dict, findings: list[dict], checkpoint_map: dict) -> int:
    """
    Posts selected findings as Drive comments on the file.
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
            success = _add_comment_to_doc(drive_service, file_id, comment_text)
        elif file_type == "google_slides":
            success = _add_comment_to_slides(drive_service, file_id, comment_text)
        else:
            success = False

        if success:
            posted += 1

    return posted
