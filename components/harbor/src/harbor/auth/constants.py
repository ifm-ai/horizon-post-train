# Modified for the RL360 source snapshot: remove private deployment and documentation references.
import os
from pathlib import Path

from harbor.constants import HARBOR_REGISTRY_WEBSITE_URL

SUPABASE_URL = os.environ.get("HARBOR_AUTH_SUPABASE_URL", "")
SUPABASE_PUBLISHABLE_KEY = os.environ.get("HARBOR_AUTH_SUPABASE_PUBLISHABLE_KEY", "")

CREDENTIALS_DIR = Path("~/.harbor").expanduser()
CREDENTIALS_PATH = CREDENTIALS_DIR / "credentials.json"
CALLBACK_PORT = 19284
HOSTED_CALLBACK_URL = f"{HARBOR_REGISTRY_WEBSITE_URL}/auth/cli-callback"
