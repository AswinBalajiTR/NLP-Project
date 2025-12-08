from __future__ import print_function
import os
import base64
import time
import pandas as pd
from email import message_from_bytes
from email.utils import parseaddr
from datetime import datetime

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from bs4 import BeautifulSoup


DEFAULT_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------

# Gmail query: Dec 6+ (because `after:2025/12/05` is strictly after Dec 5)
QUERY = "after:2025/12/05"

# Time gap between checks (seconds)
INTERVAL = 60  # 1 minute

# Excel file to update every time
import os

# Get current directory (should be .../NLP-Project copy/Code)
code_dir = os.getcwd()

# Project root is the parent of Code/
project_root = os.path.dirname(code_dir)

# Data directory lives under project root
data_dir = os.path.join(project_root, "Data")

# Final Excel path (no hardcoding)
OUTPUT_EXCEL = os.path.join(data_dir, "gmail_subject_body_date.xlsx")


class GmailLiveReader:
    def __init__(
        self,
        credentials_path="credentials.json",
        token_path="token.json",
        gmail_account_index=0,
    ):
        self.credentials_path = os.path.abspath(credentials_path)
        self.token_path = os.path.abspath(token_path)
        self.gmail_account_index = gmail_account_index
        self.gmail_web_base = f"https://mail.google.com/mail/u/{gmail_account_index}/#all/"
        self.scopes = DEFAULT_SCOPES

        self.service = self._authenticate()

    # ---------------- AUTH ----------------
    def _authenticate(self):
        creds = None

        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, self.scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.scopes
                )
                creds = flow.run_local_server(port=0)

            # Save token
            with open(self.token_path, "w") as f:
                f.write(creds.to_json())

        return build("gmail", "v1", credentials=creds)

    # ------------- UTILITIES -------------
    @staticmethod
    def html_to_text(html):
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return " ".join(soup.get_text(separator=" ", strip=True).split())

    # ------------- LIST IDS --------------
    def list_ids(self, query):
        """
        Return a list of message IDs matching the query.
        (Single-page; can extend to pagination if needed.)
        """
        ids = []
        response = self.service.users().messages().list(
            userId="me", q=query, maxResults=500, includeSpamTrash=True
        ).execute()

        ids.extend([m["id"] for m in response.get("messages", [])])
        return ids

    # ----------- FETCH ONE MAIL ----------
    def get_details(self, msg_id):
        msg = self.service.users().messages().get(
            userId="me", id=msg_id, format="raw"
        ).execute()

        raw_msg = base64.urlsafe_b64decode(msg["raw"])
        email_msg = message_from_bytes(raw_msg)

        sender_name, sender_email = parseaddr(email_msg.get("From", ""))
        subject = email_msg.get("Subject", "")
        date_raw = email_msg.get("Date", "")

        # Parse date → pretty format
        date_str = date_raw
        try:
            parsed = datetime.strptime(date_raw[:31], "%a, %d %b %Y %H:%M:%S %z")
            date_str = parsed.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

        # Extract body
        plain, html = "", ""

        if email_msg.is_multipart():
            for part in email_msg.walk():
                ctype = part.get_content_type()
                dispo = str(part.get("Content-Disposition") or "")
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True)

                if not payload:
                    continue

                if ctype == "text/plain" and "attachment" not in dispo:
                    plain += payload.decode(charset, errors="ignore")
                elif ctype == "text/html" and "attachment" not in dispo:
                    html += payload.decode(charset, errors="ignore")
        else:
            ctype = email_msg.get_content_type()
            charset = email_msg.get_content_charset() or "utf-8"
            payload = email_msg.get_payload(decode=True)
            if payload:
                if ctype == "text/plain":
                    plain = payload.decode(charset, errors="ignore")
                else:
                    html = payload.decode(charset, errors="ignore")

        body = plain if plain.strip() else self.html_to_text(html)

        return {
            "id": msg_id,
            "sender_name": sender_name,
            "sender_email": sender_email,
            "subject": subject,
            "body": body,
            "date_received": date_str,
            "gmail_link": f"{self.gmail_web_base}{msg_id}",
        }

    # -------- NEW-ONLY FETCH AS DF -------
    def fetch_new_as_dataframe(self, query, seen_ids):
        """
        - query: Gmail search string
        - seen_ids: set of already-processed message IDs (mutated in place)
        Returns a DataFrame with ONLY new emails (ids not in seen_ids).
        """
        all_ids = self.list_ids(query)
        new_ids = [mid for mid in all_ids if mid not in seen_ids]

        if not new_ids:
            return pd.DataFrame(columns=[
                "id", "sender_name", "sender_email",
                "subject", "body", "date_received", "gmail_link"
            ])

        rows = []
        for mid in new_ids:
            details = self.get_details(mid)
            rows.append(details)
            seen_ids.add(mid)   # mark as processed

        return pd.DataFrame(rows)


# -----------------------------------------------------------
# MAIN LIVE LOOP – APPEND TO SAME EXCEL
# -----------------------------------------------------------
if __name__ == "__main__":
    reader = GmailLiveReader(
        credentials_path="credentials.json",
        token_path="token.json",
        gmail_account_index=0,
    )

    # 1) Load existing Excel if present
    if os.path.exists(OUTPUT_EXCEL):
        master_df = pd.read_excel(OUTPUT_EXCEL)
        # make sure id is string
        if "id" in master_df.columns:
            seen_ids = set(master_df["id"].astype(str).tolist())
        else:
            seen_ids = set()
        print(f"[INIT] Loaded {len(master_df)} rows from {OUTPUT_EXCEL}")
    else:
        # fresh file
        master_df = pd.DataFrame(columns=[
            "id", "sender_name", "sender_email",
            "subject", "body", "date_received", "gmail_link"
        ])
        seen_ids = set()
        print(f"[INIT] No existing Excel found. Will create {OUTPUT_EXCEL}")

    print("📬 Live Gmail Reader (NEW EMAILS ONLY, APPENDING TO EXCEL)")
    print(f"Query: {QUERY}")
    print(f"Interval: {INTERVAL} seconds")
    print(f"Output Excel: {OUTPUT_EXCEL}")
    print("Press Ctrl+C to stop.\n")

    while True:
        loop_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        df_new = reader.fetch_new_as_dataframe(QUERY, seen_ids)

        if df_new.empty:
            print(f"[{loop_time}] No new emails.")
        else:
            # 2) Append to in-memory master df
            master_df = pd.concat([master_df, df_new], ignore_index=True)

            # 3) Save back to same Excel
            master_df.to_excel(OUTPUT_EXCEL, index=False)

            print(f"[{loop_time}] {len(df_new)} new email(s) appended to {OUTPUT_EXCEL}")
            print(df_new[["date_received", "sender_email", "subject"]])
            print()

        print("-" * 80)
        time.sleep(INTERVAL)
