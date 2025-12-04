#%%
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

flow = InstalledAppFlow.from_client_secrets_file("/Users/siddharthsaravanan/PycharmProjects/Pycharmgithub/EmailJobTracker/credentials.json", SCOPES)
creds = flow.run_local_server(port=0)

service = build("gmail", "v1", credentials=creds)

me = service.users().getProfile(userId="me").execute()
print("Logged in as:", me["emailAddress"])
