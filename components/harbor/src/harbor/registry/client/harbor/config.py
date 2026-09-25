# Modified for the RL360 source snapshot: remove private deployment and documentation references.
import os

from dotenv import load_dotenv

load_dotenv()

HARBOR_SUPABASE_URL = os.environ.get("HARBOR_SUPABASE_URL", "")
HARBOR_SUPABASE_PUBLISHABLE_KEY = os.environ.get("HARBOR_SUPABASE_PUBLISHABLE_KEY", "")
