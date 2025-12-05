# debug_find_workday.py

import os
import base64
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


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


def main():
    service = get_gmail_service()

    # Look for myworkday mails, no date filter
    query = "myworkday OR workday"
    results = service.users().messages().list(
        userId="me",
        q=query,
        maxResults=50
    ).execute()

    messages = results.get("messages", [])
    print(f"Found {len(messages)} messages matching query: {query}")

    for m in messages:
        full = service.users().messages().get(
            userId="me", id=m["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"]
        ).execute()

        headers = full.get("payload", {}).get("headers", [])
        msg_id = full.get("id")

        sender = ""
        subject = ""
        date = ""
        for h in headers:
            name = h.get("name", "").lower()
            if name == "from":
                sender = h.get("value", "")
            elif name == "subject":
                subject = h.get("value", "")
            elif name == "date":
                date = h.get("value", "")

        print("-" * 80)
        print("ID   :", msg_id)
        print("Date :", date)
        print("From :", sender)
        print("Subj :", subject)


if __name__ == "__main__":
    main()
