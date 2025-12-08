from __future__ import print_function
import os
import base64
import pandas as pd
from email import message_from_bytes
from email.utils import parseaddr          # <-- NEW
from datetime import datetime

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from bs4 import BeautifulSoup

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# All emails on/after 1 Dec 2025:
# "after:2025/11/30" includes 2025-12-01 and later
GMAIL_QUERY = "after:2025/08/15 before:2025/11/02"

OUTPUT_FILE = "../Data/gmail_subject_body_date.xlsx"

# Base URL to open a specific message in Gmail (account 0 / default account)
GMAIL_WEB_BASE = "https://mail.google.com/mail/u/0/#all/"

# None = no hard cap (fetch everything matching query)
# or set to e.g. 1000 if you want a limit
MAX_RESULTS = None


# -----------------------------------------
# HELPERS
# -----------------------------------------
def clean_for_excel(text, max_len=32000):
    """Clean text so Excel doesn't treat it as a bad formula or invalid string."""
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)

    # Remove control characters (except newline and tab)
    cleaned = []
    for ch in text:
        code = ord(ch)
        if ch in ("\n", "\t") or code >= 32:
            cleaned.append(ch)
    text = "".join(cleaned)

    # Avoid Excel formula interpretation: if it starts with = + - @, prefix with '
    if text.startswith(("=", "+", "-", "@")):
        text = "'" + text

    # Truncate very long strings to avoid Excel cell limits
    if len(text) > max_len:
        text = text[:max_len]

    return text


def html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style tags
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Get visible text
    text = soup.get_text(separator=" ", strip=True)
    # Collapse multiple spaces
    return " ".join(text.split())


# AUTHENTICATION
def get_gmail_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


# FETCH *ALL* MESSAGE IDS WITH PAGINATION
def list_all_message_ids(service, max_results=None, query=""):
    """
    Return list of Gmail message IDs matching the query.
    Uses pagination to go through all pages.
    - max_results: if None, fetch everything; else stop when limit reached.
    """
    all_ids = []
    page_token = None

    while True:
        page_size = 500  # max per API call
        if max_results is not None:
            remaining = max_results - len(all_ids)
            if remaining <= 0:
                break
            page_size = min(page_size, remaining)

        response = service.users().messages().list(
            userId="me",
            q=query,
            maxResults=page_size,
            pageToken=page_token,
            includeSpamTrash=True,  # include spam & trash as well
        ).execute()

        messages = response.get("messages", [])
        all_ids.extend([m["id"] for m in messages])

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    print(f"Total message IDs fetched: {len(all_ids)}")
    return all_ids


# FETCH DETAILS (SENDER, SUBJECT, BODY, DATE)
def get_message_details(service, msg_id):
    """Get sender_name, sender_email, subject, body (plain text), and date from a Gmail message."""
    msg = service.users().messages().get(
        userId="me", id=msg_id, format="raw"
    ).execute()

    # Decode raw email
    raw_msg = base64.urlsafe_b64decode(msg["raw"].encode("UTF-8"))
    email_msg = message_from_bytes(raw_msg)

    # ----- Sender -----
    from_header = email_msg.get("From", "") or ""
    sender_name, sender_email = parseaddr(from_header)

    # Subject
    subject = email_msg.get("Subject", "") or ""

    # Date header
    date_raw = email_msg.get("Date", "") or ""
    date_str = date_raw
    try:
        # Common format: "Fri, 05 Dec 2025 15:42:18 +0000"
        parsed = datetime.strptime(date_raw[:31], "%a, %d %b %Y %H:%M:%S %z")
        date_str = parsed.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        # keep raw if parsing fails
        pass

    # --- Extract body ---
    plain_body = ""
    html_body = ""

    if email_msg.is_multipart():
        for part in email_msg.walk():
            content_type = part.get_content_type()
            content_dispo = str(part.get("Content-Disposition") or "")
            charset = part.get_content_charset() or "utf-8"
            payload = part.get_payload(decode=True)

            if not payload:
                continue

            if content_type == "text/plain" and "attachment" not in content_dispo:
                plain_body += payload.decode(charset, errors="ignore")
            elif content_type == "text/html" and "attachment" not in content_dispo:
                html_body += payload.decode(charset, errors="ignore")
    else:
        content_type = email_msg.get_content_type()
        charset = email_msg.get_content_charset() or "utf-8"
        payload = email_msg.get_payload(decode=True)
        if payload:
            if content_type == "text/plain":
                plain_body = payload.decode(charset, errors="ignore")
            elif content_type == "text/html":
                html_body = payload.decode(charset, errors="ignore")

    # Prefer plain text, fall back to HTML→text
    if plain_body.strip():
        body = plain_body
    else:
        body = html_to_text(html_body)

    # Clean for Excel
    sender_name = clean_for_excel(sender_name)
    sender_email = clean_for_excel(sender_email)
    subject = clean_for_excel(subject)
    body = clean_for_excel(body)
    date_str = clean_for_excel(date_str)

    return sender_name, sender_email, subject, body, date_str


# -----------------------------------------
# MAIN: SAVE TO EXCEL
# -----------------------------------------
def save_gmail_to_excel(
    excel_file=OUTPUT_FILE, max_results=MAX_RESULTS, query=GMAIL_QUERY
):
    """Fetch emails and save id, sender_name, sender_email, subject, body, date_received, gmail_link into an Excel file."""
    service = get_gmail_service()
    message_ids = list_all_message_ids(service, max_results=max_results, query=query)

    print(f"Using query: {query!r}")
    print(f"Number of messages to process: {len(message_ids)}")

    data = []
    for i, mid in enumerate(message_ids, start=1):
        sender_name, sender_email, subject, body, date_received = get_message_details(
            service, mid
        )

        # Build direct Gmail URL for this message
        gmail_link = f"{GMAIL_WEB_BASE}{mid}"

        data.append(
            {
                "id": mid,
                "sender_name": sender_name,
                "sender_email": sender_email,
                "subject": subject,
                "body": body,
                "date_received": date_received,
                "gmail_link": gmail_link,   # <-- NEW COLUMN
            }
        )
        if i % 50 == 0:
            print(f"Processed {i} messages...")

    df = pd.DataFrame(data)
    df.to_excel(excel_file, index=False)
    print(f"[✓] Saved {len(df)} emails to {excel_file}")


if __name__ == "__main__":
    save_gmail_to_excel()
