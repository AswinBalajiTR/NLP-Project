import os
import re
import base64
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
import joblib
from bs4 import BeautifulSoup
from pandas.errors import EmptyDataError

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# CONFIG
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
MODEL_PATH = "models/job_email_model.joblib"

# DATA MODEL
@dataclass
class SimpleEmail:
    email_id: str
    date: str
    sender: str
    subject: str
    body: str

# GMAIL AUTH
def get_gmail_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def get_header(headers: List[Dict[str, Any]], name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def extract_body(payload: Dict[str, Any]) -> str:
    """
    Try text/plain; if missing, fallback to text/html and strip tags.
    """
    if "parts" in payload:
        # text/plain first
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain":
                data = part["body"].get("data")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        # then text/html
        for part in payload["parts"]:
            if part.get("mimeType") == "text/html":
                data = part["body"].get("data")
                if data:
                    html = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                    soup = BeautifulSoup(html, "html.parser")
                    return soup.get_text(separator=" ", strip=True)
    else:
        data = payload.get("body", {}).get("data")
        if data:
            decoded = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
            if "<html" in decoded.lower():
                soup = BeautifulSoup(decoded, "html.parser")
                return soup.get_text(separator=" ", strip=True)
            return decoded
    return ""


# ==============================================================
# TEXT HELPERS
# ==============================================================

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", " ", text)
    return text.strip()


def combine_fields(subject: str, body: str, sender: str) -> str:
    return f"SUBJ: {clean_text(subject)} SENDER: {clean_text(sender)} BODY: {clean_text(body)}"


def infer_company_from_sender(sender: str) -> str:
    sender = (sender or "").lower()
    match = re.search(r"@([a-z0-9\-\.]+)", sender)
    if not match:
        return ""
    domain = match.group(1)
    domain = domain.replace("mail.", "")
    parts = domain.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return parts[0]


# ==============================================================
# STRONG JOB SIGNALS (HYBRID: RULES + ML)
# ==============================================================

JOB_DOMAINS_INFER = [
    "indeed",
    "linkedin",
    "handshake",
    "joinhandshake",
    "workday",
    "myworkday",
    "greenhouse",
    "bamboohr",
    "icims",
    "smartrecruiters",
    "lever.co",
    "jazzhr",
    "monster",
    "careerbuilder",
    "workable",
    "eightfold",
]

STRONG_SUBJECT_INFER = [
    "application received",
    "application submitted",
    "application update",
    "thank you for applying",
    "your application",
    "job application",
    "we received your application",
    "we have received your application",
    "interview invitation",
    "invitation to interview",
    "schedule your interview",
    "job offer",
    "offer letter",
]

STRONG_BODY_INFER = [
    # Workday-style confirmations like what you pasted:
    "this email is to serve as confirmation that we are in receipt of your application",
    "confirmation that we are in receipt of your application",
    "we are in receipt of your application",
    "we have received your application",
    "your application has been received",
    "thank you for applying",
    "your application is currently being reviewed",
    "a dedicated member of our talent acquisition team will be in contact",
    "a member of our talent acquisition team will be in contact",
    "a recruiter will be in contact",
    "we regret to inform you",
    "we have decided not to move forward with your application",
    "schedule an interview",
    "interview invitation",
    "invitation to interview",
]


def has_strong_job_signal(subject: str, body: str, sender: str) -> bool:
    """
    Return True if this email is *obviously* job-related, regardless of ML score.
    Example: Workday/LinkedIn/Handshake confirmations, clear application/offer language.
    """
    subj = subject.lower() if isinstance(subject, str) else ""
    text = f"{subject} [SEP] {body}".lower()
    sender_lower = sender.lower() if isinstance(sender, str) else ""

    # 1) Known job platforms / ATS vendors
    if any(dom in sender_lower for dom in JOB_DOMAINS_INFER):
        return True

    # 2) Strong phrases in subject
    if any(p in subj for p in STRONG_SUBJECT_INFER):
        return True

    # 3) Strong phrases in body
    if any(p in text for p in STRONG_BODY_INFER):
        return True

    return False


# ==============================================================
# STATUS RULES (APPLICATION / INTERVIEW / REJECTION)
# ==============================================================

APPLICATION_CONFIRM_WORDS = [
    "application received",
    "thank you for applying",
    "we have received your application",
    "your application has been received",
    "this email is to serve as confirmation that we are in receipt of your application",
    "we are in receipt of your application",
    "your application is currently being reviewed",
]

INTERVIEW_WORDS = [
    "schedule an interview",
    "interview invitation",
    "invitation to interview",
    "virtual interview",
    "onsite interview",
    "phone interview",
]

REJECTION_WORDS = [
    "we regret to inform you",
    "not moving forward with your application",
    "we have decided not to move forward",
    "your application was not selected",
    "we will not be moving forward",
]


def infer_job_status(text: str) -> str:
    t = text.lower()

    if any(kw in t for kw in APPLICATION_CONFIRM_WORDS):
        return "APPLICATION_CONFIRMATION"

    if any(kw in t for kw in INTERVIEW_WORDS):
        return "INTERVIEW_INVITE"

    if any(kw in t for kw in REJECTION_WORDS):
        return "REJECTION"

    return "OTHER_JOB_EMAIL"


def applied_flag_from_status(status: str) -> int:
    return 1 if status in ["APPLICATION_CONFIRMATION", "INTERVIEW_INVITE", "REJECTION"] else 0


# ==============================================================
# ML-BASED INBOX AGENT (HYBRID INFERENCE, v2 WITH DEBUG)
# ==============================================================

class InboxAgentGmailML:

    def __init__(self, service, model_path: str = MODEL_PATH, prob_threshold: float = 0.2):
        self.service = service
        self.prob_threshold = prob_threshold

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                f"Run auto_train_job_email_model.py first."
            )

        self.model = joblib.load(model_path)
        print("Model loaded from", os.path.abspath(model_path))

    def fetch_emails(self, query: str = "", max_results: int = 2000) -> List[SimpleEmail]:
        """
        v2: By default, query="" (no filter) and higher max_results.
        Also logs any Workday/ATS emails that appear.
        """
        results = self.service.users().messages().list(
            userId="me", q=query, maxResults=max_results
        ).execute()

        emails: List[SimpleEmail] = []
        for m in results.get("messages", []):
            full = self.service.users().messages().get(
                userId="me", id=m["id"], format="full"
            ).execute()

            payload = full.get("payload", {})
            headers = payload.get("headers", [])

            email = SimpleEmail(
                email_id=m["id"],
                date=get_header(headers, "Date"),
                sender=get_header(headers, "From"),
                subject=get_header(headers, "Subject"),
                body=extract_body(payload),
            )

            # DEBUG: Log any Workday/ATS-like sender
            sender_lower = (email.sender or "").lower()
            if any(dom in sender_lower for dom in ["workday", "myworkday", "greenhouse", "handshake", "joinhandshake", "linkedin"]):
                print("\nATS email detected in fetch_emails():")
                print("  ID   :", email.email_id)
                print("  Date :", email.date)
                print("  From :", email.sender)
                print("  Subj :", email.subject)

            emails.append(email)

        print(f"Fetched {len(emails)} emails from Gmail.")
        return emails

    def process_and_update_csv(self, emails, master_csv_path: str = "job_emails_from_gmail_ml_v2.csv"):
        rows = []

        # Run model on all emails
        texts = [combine_fields(e.subject, e.body, e.sender) for e in emails]
        probas = self.model.predict_proba(texts)

        # Debug: how many emails above different probability cutoffs
        total = len(probas)
        above_05 = sum(float(p[1]) >= 0.5 for p in probas)
        above_03 = sum(float(p[1]) >= 0.3 for p in probas)
        above_02 = sum(float(p[1]) >= 0.2 for p in probas)
        print(f"[Debug v2] total={total}, p>=0.5: {above_05}, p>=0.3: {above_03}, p>=0.2: {above_02}")

        for email, p in zip(emails, probas):
            p_job = float(p[1])

            # Hybrid logic: ML probability OR strong rule-based job signal
            strong = has_strong_job_signal(email.subject, email.body, email.sender)

            # DEBUG: log Workday/ATS email classification decision
            sender_lower = (email.sender or "").lower()
            if any(dom in sender_lower for dom in ["workday", "myworkday", "greenhouse", "handshake", "joinhandshake", "linkedin"]):
                print("\n[PROCESS DEBUG v2] ATS email encountered in process_and_update_csv():")
                print("  ID      :", email.email_id)
                print("  From    :", email.sender)
                print("  Subject :", email.subject)
                print("  p_job   :", p_job)
                print("  strong  :", strong)

            # If not obviously job-related AND below threshold → skip
            if (not strong) and (p_job < self.prob_threshold):
                if any(dom in sender_lower for dom in ["workday", "myworkday", "greenhouse", "handshake", "joinhandshake", "linkedin"]):
                    print("  -> SKIPPED ATS email due to low p_job and strong=False")
                continue

            combined = clean_text(email.subject) + " [SEP] " + clean_text(email.body)
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
                "job_prob": p_job,
                "job_status": status,
                "applied": applied
            })

        new_df = pd.DataFrame(rows)
        abs_path = os.path.abspath(master_csv_path)

        # If NO job emails found in this run
        if new_df.empty:
            if os.path.exists(master_csv_path) and os.path.getsize(master_csv_path) > 0:
                try:
                    old_df = pd.read_csv(master_csv_path)
                    print(f"No new job emails. Master has {len(old_df)} rows.")
                    print(f"Master CSV path: {abs_path}")
                    return old_df
                except EmptyDataError:
                    print("Master CSV exists but is empty/corrupt; recreating.")
                    new_df.to_csv(master_csv_path, index=False)
                    print(f"Created new master CSV (0 rows). Path: {abs_path}")
                    return new_df
            else:
                new_df.to_csv(master_csv_path, index=False)
                print(f"Created new master CSV (0 rows). Path: {abs_path}")
                return new_df

        # We DO have new job emails this run
        if os.path.exists(master_csv_path) and os.path.getsize(master_csv_path) > 0:
            try:
                old_df = pd.read_csv(master_csv_path)
            except EmptyDataError:
                print("Master CSV exists but is empty/corrupt; ignoring old content.")
                old_df = pd.DataFrame()

            before = len(old_df)
            merged = pd.concat([old_df, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["email_id"], keep="last")
            after = len(merged)
            added = after - before
            merged.to_csv(master_csv_path, index=False)
            print(f"Existing rows: {before}, New: {len(new_df)}, Added: {added}")
            print(f"Master CSV path: {abs_path}")
            return merged
        else:
            # Master doesn't exist or 0 bytes → just create it
            new_df.to_csv(master_csv_path, index=False)
            print(f"Created new master CSV with {len(new_df)} rows.")
            print(f"Master CSV path: {abs_path}")
            return new_df
