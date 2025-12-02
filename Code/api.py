def get_job_emails(service):
    query = "subject:(application) OR subject:(interview) OR from:(linkedin.com OR indeed.com OR greenhouse.io)"
    
    results = service.users().messages().list(
        userId='me',
        q=query
    ).execute()

    messages = results.get('messages', [])

    for msg in messages:
        msg_data = service.users().messages().get(
            userId='me', id=msg['id'], format='full'
        ).execute()

        headers = msg_data['payload']['headers']
        body = msg_data['snippet']

        print("\n------------------------")
        for h in headers:
            if h['name'] in ('From', 'Subject', 'Date'):
                print(f"{h['name']}: {h['value']}")
        print("Body:", body)


if __name__ == "__main__":
    service = get_gmail_service()
    get_job_emails(service)
