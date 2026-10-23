
import os

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()


def get_supabase_client() -> Client:
    """Create a Supabase client using local environment variables."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be configured in .env"
        )

    return create_client(url, key)


if __name__ == "__main__":
    try:
        client = get_supabase_client()

        # Read-only connectivity test.
        response = (
            client.table("documents")
            .select("id")
            .limit(1)
            .execute()
        )

        print("Supabase connection successful.")
        print(f"Documents table is accessible: {response.data is not None}")

    except Exception as exc:
        print(f"Supabase connection failed: {type(exc).__name__}: {exc}")
