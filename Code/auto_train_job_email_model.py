# auto_train_job_email_model.py

import os
import re
import base64
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
from bs4 import BeautifulSoup

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


# ==============================================================
# CONFIG
# ==============================================================

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
MODEL_PATH = "models/job_email_model.joblib"
os.makedirs("models", exist_ok=True)


# ==============================================================
# GMAIL HELPERS
# ==============================================================

@dataclass
class SimpleEmail:
    email_id: str
    date: str
    sender: str
    subject: str
    body: str


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


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", " ", text)
    return text.strip()


def combine_fields(subject: str, body: str, sender: str) -> str:
    return f"SUBJ: {clean_text(subject)} SENDER: {clean_text(sender)} BODY: {clean_text(body)}"


# ==============================================================
# DOMAIN LISTS
# ==============================================================

# Job-related domains (we treat these as positive sources)
JOB_DOMAINS = [
    "indeed",
    "linkedin",
    "handshake",
    "joinhandshake",
    "ziprecruiter",
    "glassdoor",
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
    "nexxt",
    "lensa",
    "jobfalcon",
    "careerpi",
    "aptena",
]

# Domains we KNOW are NOT job emails (banks, coffee, shopping, etc.)
NEGATIVE_DOMAINS = [
    "sofi",
    "bankofamerica",
    "bofa",
    "chase",
    "wellsfargo",
    "capitalone",
    "americanexpress",
    "amex",
    "starbucks",
    "ubereats",
    "doordash",
    "grubhub",
    "mcdonalds",
    "burgerking",
    "dominos",
    "netflix",
    "spotify",
    "apple.com",
    "appleid",
    "primevideo",
    "hulu",
    "disneyplus",
    "amazon",
    "amzn"
]


def is_probably_job_sender(sender: str) -> bool:
    s = sender.lower()
    return any(dom in s for dom in JOB_DOMAINS)


def is_negative_sender(sender: str) -> bool:
    s = sender.lower()
    return any(dom in s for dom in NEGATIVE_DOMAINS)


# ==============================================================
# WEAK-SUPERVISION LABELING RULE
# ==============================================================

def is_job_email_rule(subject: str, body: str, sender: str) -> int:
    """
    Weak-supervision rule:
    returns 1 if we *treat* this as job-related for training, else 0.
    This is ONLY for creating labels for the model.
    """

    sender_lower = sender.lower() if isinstance(sender, str) else ""
    subj = subject.lower() if isinstance(subject, str) else ""
    text = f"{subject} [SEP] {body}".lower()

    # 0) HARD NEGATIVE: banks, coffee, streaming, generic consumer crap
    if is_negative_sender(sender_lower):
        return 0

    STRONG_SUBJECT = [
        "your application",
        "application received",
        "application update",
        "application submitted",
        "thank you for applying",
        "we received your application",
        "we have received your application",
        "interview",
        "interview invitation",
        "virtual interview",
        "onsite interview",
        "next steps",
        "offer letter",
        "job offer",
        "role:",
        "position:",
        "position -",
        "position ",
    ]

    STRONG_BODY = [
        "your application has been received",
        "we have received your application",
        "this is to confirm your application",
        "thank you for applying",
        "we regret to inform you",
        "we have decided not to move forward",
        "schedule an interview",
        "interview invitation",
        "invitation to interview",
    ]

    # 1) Strong subject phrase → job email
    if any(p in subj for p in STRONG_SUBJECT):
        return 1

    # 2) Strong body phrase → job email
    if any(p in text for p in STRONG_BODY):
        return 1

    # 3) If sender is from known job platform (LinkedIn, Handshake, Workday, Greenhouse etc.)
    if is_probably_job_sender(sender_lower):
        # For training, treat ALL of these as job-related emails
        return 1

    # 4) Everything else treated as non-job for training
    return 0


# ==============================================================
# TRAINING
# ==============================================================

def auto_train_model(max_train_emails: int = 800):
    service = get_gmail_service()
    print("[AutoTrain] Fetching emails for weakly supervised training...")

    results = service.users().messages().list(
        userId="me",
        q="newer_than:180d",          # last ~6 months
        maxResults=max_train_emails
    ).execute()

    messages = results.get("messages", [])
    if not messages:
        raise RuntimeError("[AutoTrain] No emails fetched.")

    rows = []
    for m in messages:
        full = service.users().messages().get(
            userId="me", id=m["id"], format="full"
        ).execute()

        payload = full.get("payload", {})
        headers = payload.get("headers", [])

        subject = get_header(headers, "Subject")
        sender = get_header(headers, "From")
        body = extract_body(payload)

        label = is_job_email_rule(subject, body, sender)

        rows.append({
            "email_id": m["id"],
            "sender": sender,
            "subject": clean_text(subject),
            "body": clean_text(body),
            "is_job_email": label
        })

    df = pd.DataFrame(rows)

    # Split into positives and negatives
    pos = df[df["is_job_email"] == 1]
    neg = df[df["is_job_email"] == 0]

    if len(pos) == 0:
        raise RuntimeError("[AutoTrain] No positive job emails found by rules; cannot train.")

    # Sample up to 3x as many negatives as positives to avoid huge imbalance
    neg_sample = neg.sample(min(len(neg), len(pos) * 3), random_state=42)
    df_bal = pd.concat([pos, neg_sample], ignore_index=True)

    df_bal["text"] = df_bal.apply(
        lambda row: combine_fields(row["subject"], row["body"], row["sender"]), axis=1
    )

    X = df_bal["text"].tolist()
    y = df_bal["is_job_email"].tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=-1)),
    ])

    print(f"[AutoTrain] Training on {len(X_train)} examples, validating on {len(X_val)}...")
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"[AutoTrain] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    auto_train_model()
