from google.cloud import bigquery
from google.oauth2 import service_account

KEY_PATH = (
    "/Users/steinerabayie/Desktop/genai/cosmic-kayak-462516-q8-b5d1fed3c8df_copy.json"
)
creds = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(project="cosmic-kayak-462516-q8", credentials=creds)

# Test listing tables
tables = list(client.list_tables("test_cap"))
print(f"Tables in test_cap: {[t.table_id for t in tables]}")
