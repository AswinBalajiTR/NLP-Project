from inbox_agent_gmail import get_gmail_service, InboxAgentGmail


def main():
    service = get_gmail_service()
    agent = InboxAgentGmail(service)

    # Fetch last 90 days
    emails = agent.fetch_emails(query="newer_than:90d")

    # Process → Save CSV
    df = agent.process_emails_to_csv(emails, "job_emails_from_gmail.csv")

    print(df.head())
    print("\n[Summary]")
    print(df["job_status"].value_counts())
    print("\nDone.")


if __name__ == "__main__":
    main()
