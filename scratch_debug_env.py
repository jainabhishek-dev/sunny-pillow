import os
from dotenv import load_dotenv
load_dotenv()
print(f"ID: '{os.getenv('GOOGLE_CLIENT_ID')}'")
print(f"SECRET: '{os.getenv('GOOGLE_CLIENT_SECRET')}'")
