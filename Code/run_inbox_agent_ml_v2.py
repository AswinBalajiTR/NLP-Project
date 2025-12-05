import os
import traceback
from inbox_agent_gmail_ml_v2 import get_gmail_service, InboxAgentGmailML


def main():
    print("[RUN V2] Starting run_inbox_agent_ml_v2.py")
    print("Current working directory:", os.getcwd())

    service = get_gmail_service()
    agent = InboxAgentGmailML(service, prob_threshold=0.2)

    print("Fetching emails (query='', max_results=2000)...")
    emails = agent.fetch_emails(query="", max_results=2000)
    print(f"Fetched {len(emails)} emails.")

    csv_path = os.path.join(os.getcwd(), "job_emails_from_gmail_ml_v2.csv")
    print("[RUN V2] CSV will be written to:", csv_path)

    df = agent.process_and_update_csv(
        emails,
        master_csv_path="job_emails_from_gmail_ml_v2.csv"
    )

    print("\n[RUN V2] HEAD OF RESULT DF")
    print(df.head())

    if df.empty or "job_status" not in df.columns:
        print("\n[RUN V2] Summary: No job emails found OR no 'job_status' column.")
    else:
        print("\n[RUN V2] Summary by status:")
        print(df["job_status"].value_counts())

    print("\n[RUN V2] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[RUN V2] ERROR OCCURRED")
        print(type(e).__name__, ":", e)
        print("Full traceback:")
        traceback.print_exc()
