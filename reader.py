import re
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from io import BytesIO

# MIME types the app supports
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
    """Extract Google Drive file ID from a share link."""
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


def _get_pdf_bytes_by_id(file_id: str, file_type: str, drive_service) -> bytes:
    """Get PDF bytes for a file by ID and type."""
    if file_type in ("google_doc", "google_slides"):
        # Export Docs/Slides as PDF
        pdf_bytes = (
            drive_service.files()
            .export(fileId=file_id, mimeType="application/pdf")
            .execute()
        )
    else:  # file_type == "pdf"
        # Download PDF directly
        pdf_bytes = (
            drive_service.files()
            .get_media(fileId=file_id)
            .execute()
        )
    return pdf_bytes


def get_file_as_pdf(token: dict, drive_url: str) -> dict:
    """
    Exports a Google Drive file as PDF bytes.

    Supports:
    - Google Docs: exported as PDF via Drive API
    - Google Slides: exported as PDF via Drive API
    - PDF files: downloaded directly

    Returns a dict with:
        file_id    - the Drive file ID
        title      - the document title
        file_type  - 'google_doc' | 'google_slides' | 'pdf'
        pdf_bytes  - raw PDF file bytes
    """
    file_id = extract_file_id(drive_url)
    creds = _build_credentials(token)
    drive_service = _build_drive_service(creds)

    # Get file metadata
    metadata = (
        drive_service.files()
        .get(fileId=file_id, fields="id,name,mimeType", supportsAllDrives=True)
        .execute()
    )
    mime_type = metadata.get("mimeType", "")
    title = metadata.get("name", "Untitled")

    # Check for unsupported types
    if mime_type in UNSUPPORTED_TYPES:
        raise ValueError(UNSUPPORTED_TYPES[mime_type])

    if mime_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"This file type ({mime_type}) is not supported. "
            "Please provide a link to a Google Doc, Google Slides presentation, or PDF."
        )

    file_type = SUPPORTED_TYPES[mime_type]

    # Export as PDF
    pdf_bytes = _get_pdf_bytes_by_id(file_id, file_type, drive_service)

    if not pdf_bytes or len(pdf_bytes) == 0:
        raise ValueError(
            f"No content was retrieved from '{title}'. "
            "The file may be empty or you may not have access to it."
        )

    return {
        "file_id": file_id,
        "title": title,
        "file_type": file_type,
        "pdf_bytes": pdf_bytes,
    }


def create_drive_subfolder(token: dict, parent_folder_id: str, folder_name: str) -> str:
    """Create a subfolder inside parent_folder_id. Returns the new folder's Drive file ID."""
    creds = _build_credentials(token)
    drive_service = _build_drive_service(creds)
    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    folder = drive_service.files().create(body=metadata, fields="id").execute()
    return folder["id"]


def upload_jpeg_to_drive(token: dict, folder_id: str, filename: str, image_bytes: bytes) -> str:
    """Upload a JPEG image to a Drive folder. Returns the uploaded file's Drive file ID."""
    from googleapiclient.http import MediaIoBaseUpload
    creds = _build_credentials(token)
    drive_service = _build_drive_service(creds)
    metadata = {"name": filename, "parents": [folder_id]}
    media = MediaIoBaseUpload(BytesIO(image_bytes), mimetype="image/jpeg")
    file = drive_service.files().create(
        body=metadata, media_body=media, fields="id"
    ).execute()
    return file["id"]


def get_pdf_bytes_by_id(token: dict, file_id: str) -> dict:
    """
    Get PDF bytes for a file using its ID (without needing to parse a URL).
    Used by the SSE stream endpoint to get PDF bytes during job processing.

    Returns a dict with:
        file_type  - 'google_doc' | 'google_slides' | 'pdf'
        pdf_bytes  - raw PDF file bytes
    """
    creds = _build_credentials(token)
    drive_service = _build_drive_service(creds)

    # Get file metadata
    metadata = (
        drive_service.files()
        .get(fileId=file_id, fields="mimeType", supportsAllDrives=True)
        .execute()
    )
    mime_type = metadata.get("mimeType", "")

    if mime_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"This file type ({mime_type}) is not supported."
        )

    file_type = SUPPORTED_TYPES[mime_type]

    # Export as PDF
    pdf_bytes = _get_pdf_bytes_by_id(file_id, file_type, drive_service)

    if not pdf_bytes or len(pdf_bytes) == 0:
        raise ValueError(
            f"No content was retrieved. "
            "The file may be empty or you may not have access to it."
        )

    return {
        "file_type": file_type,
        "pdf_bytes": pdf_bytes,
    }
