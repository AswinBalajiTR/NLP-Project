# inbox_agent_gmail.py

import os
import re
import base64
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# 1. Gmail API Setup
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def get_gmail_service():
    """
    Uses credentials.json + token.json to authenticate Gmail API.
    """
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)

# 2. Email Data Model
@dataclass
class SimpleEmail:
    email_id: str
    date: str
    sender: str
    subject: str
    body: str

# 3. Helper Functions
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", " ", text)
    return text.strip()


def combine_subject_body(subject: str, body: str) -> str:
    subject = clean_text(subject)
    body = clean_text(body)
    return f"{subject} [SEP] {body}".lower()

# 4. Job Email Detection Rules
JOB_DOMAINS = [
    "indeed", "linkedin", "ziprecruiter", "glassdoor", "workday",
    "myworkday", "bamboohr", "lever.co", "greenhouse", "icims",
    "smartrecruiters", "jazzhr", "monster", "careerbuilder",
    "workable", "eightfold", "myjobhelper", "talenthub"
]

# Strong subject indicators
STRONG_SUBJECT = [
    "your application",
    "application received",
    "application update",
    "application submitted",
    "thank you for applying",
    "interview",
    "interview invitation",
    "next steps",
    "we are hiring",
    "we're hiring",
    "job opening",
    "job opportunity",
    "position",
    "role:",
    "now hiring",
    "job for you",
]

# Strong body phrases
STRONG_BODY = [
    "your application has been received",
    "this is to confirm your application",
    "thank you for applying",
    "we regret to inform you",
    "we have decided not to move forward",
    "interview invitation",
    "schedule an interview",
]


def is_probably_job_sender(sender: str) -> bool:
    s = sender.lower()
    return any(dom in s for dom in JOB_DOMAINS)


def is_job_email(subject: str, body: str, sender: str) -> bool:
    subj = subject.lower() if isinstance(subject, str) else ""
    text = f"{subject} [SEP] {body}".lower()

    # 1 — Strong subject patterns
    if any(p in subj for p in STRONG_SUBJECT):
        return True

    # 2 — Recognized job platform sender + job-ish subject
    if is_probably_job_sender(sender):
        if any(word in subj for word in ["job", "role", "position", "application", "hiring"]):
            return True

    # 3 — Strong body patterns
    if any(p in text for p in STRONG_BODY):
        return True

    return False


# Status detection
APPLICATION_CONFIRM_WORDS = [
    "application received", "thank you for applying",
    "we have received your application", "application submitted",
]

INTERVIEW_WORDS = [
    "schedule an interview", "interview invitation",
    "interview", "next steps", "virtual interview",
]

REJECTION_WORDS = [
    "we regret to inform you", "not moving forward",
    "unfortunately", "your application was not selected",
]


def infer_job_status(text: str) -> str:
    t = text.lower()

    if any(kw in t for kw in APPLICATION_CONFIRM_WORDS):
        return "APPLICATION_CONFIRMATION"

    if any(kw in t for kw in INTERVIEW_WORDS):
        return "INTERVIEW_INVITE"

    if any(kw in t for kw in REJECTION_WORDS):
        return "REJECTION"

    if any(kw in t for kw in STRONG_SUBJECT):
        return "JOB_OPENING"

    return "OTHER_JOB_EMAIL"


def applied_flag_from_status(status: str) -> int:
    return 1 if status in ["APPLICATION_CONFIRMATION", "INTERVIEW_INVITE", "REJECTION"] else 0


def infer_company_from_sender(sender: str) -> str:
    sender = sender.lower()
    match = re.search(r"@([a-z0-9\-\.]+)", sender)
    if not match:
        return ""
    domain = match.group(1)
    domain = domain.replace("mail.", "")
    parts = domain.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return parts[0]

# 5. Gmail Body Extraction
def extract_body(payload: Dict[str, Any]) -> str:
    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain":
                data = part["body"].get("data")
                if data:
                    try:
                        return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                    except:
                        pass

    # Fallback
    data = payload.get("body", {}).get("data")
    if data:
        try:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        except:
            return ""

    return ""


def get_header(headers: List[Dict[str, Any]], name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""

# 6. MAIN AGENT
class InboxAgentGmail:

    def __init__(self, service):
        self.service = service

    def fetch_emails(self, query="newer_than:90d", max_results=500) -> List[SimpleEmail]:
        messages = self.service.users().messages().list(
            userId="me", q=query, maxResults=max_results
        ).execute()

        email_list = []
        for m in messages.get("messages", []):
            full = self.service.users().messages().get(
                userId="me", id=m["id"], format="full"
            ).execute()

            payload = full.get("payload", {})
            headers = payload.get("headers", [])

            email_list.append(
                SimpleEmail(
                    email_id=m["id"],
                    date=get_header(headers, "Date"),
                    sender=get_header(headers, "From"),
                    subject=get_header(headers, "Subject"),
                    body=extract_body(payload)
                )
            )

        return email_list

    def process_emails_to_csv(self, emails, output_csv_path="job_emails_from_gmail.csv"):
        rows = []

        for email in emails:
            if not is_job_email(email.subject, email.body, email.sender):
                continue

            combined = combine_subject_body(email.subject, email.body)
            status = infer_job_status(combined)
            applied = applied_flag_from_status(status)
            company = infer_company_from_sender(email.sender)

            rows.append({
                "email_id": email.email_id,
                "date": email.date,
                "sender": email.sender,
                "company": company,
                "subject": clean_text(email.subject),
                "body": clean_text(email.body),
                "job_status": status,
                "applied": applied
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_csv_path, index=False)
        print(f"[InboxAgent] Saved {len(df)} job emails → {output_csv_path}")
        return df
