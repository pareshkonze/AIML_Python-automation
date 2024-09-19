import asyncio
import os
import sys

# Add the parent directory to sys.path to allow imports from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from review_fetcher.main import handle_review_fetch
from review_analyzer.main import handle_review_analysis
from common.database import close_db_pool

async def run_services():
    try:
        print("Running Review Fetcher (Google)...")
        google_result = await handle_review_fetch({'source': 'google'}, None)
        print(f"Google Review Fetcher Result: {google_result}")

        print("\nRunning Review Fetcher (Facebook)...")
        facebook_result = await handle_review_fetch({'source': 'facebook'}, None)
        print(f"Facebook Review Fetcher Result: {facebook_result}")

        print("\nRunning Review Analyzer...")
        analyzer_result = await handle_review_analysis(None, None)
        print(f"Review Analyzer Result: {analyzer_result}")
    finally:
        await close_db_pool()

if __name__ == "__main__":
    asyncio.run(run_services())