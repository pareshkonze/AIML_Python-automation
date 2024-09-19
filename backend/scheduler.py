import os
import sys
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import asyncio
import logging

# Add the parent directory to sys.path to allow imports from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from review_fetcher.main import handle_review_fetch
from review_analyzer.main import handle_review_analysis
from common.database import init_db_pool, close_db_pool

# Update the path to .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Set up logging
logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), '..', 'logs', 'scheduler.log'), 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_review_fetch_and_analyze():
    try:
        await init_db_pool()
        
        # Fetch Google reviews
        logger.info("Starting Google review fetch...")
        google_result = await handle_review_fetch({'source': 'google'}, None)
        logger.info(f"Google Review Fetch Result: {google_result}")

        # Fetch Facebook reviews
        logger.info("Starting Facebook review fetch...")
        facebook_result = await handle_review_fetch({'source': 'facebook'}, None)
        logger.info(f"Facebook Review Fetch Result: {facebook_result}")

        # Analyze reviews
        logger.info("Starting review analysis...")
        analyzer_result = await handle_review_analysis(None, None)
        logger.info(f"Review Analysis Result: {analyzer_result}")

    except Exception as e:
        logger.error(f"Error in review fetch and analyze process: {str(e)}", exc_info=True)
    finally:
        await close_db_pool()

def start_scheduler():
    scheduler = AsyncIOScheduler()
    
    # Schedule the job to run at the specified interval (Every 24 hours)
    interval_minutes = int(os.getenv('SCHEDULE_INTERVAL_MINUTES', '1440'))
    scheduler.add_job(
        run_review_fetch_and_analyze,
        trigger=IntervalTrigger(minutes=interval_minutes),
        id='review_fetch_and_analyze',
        name='Fetch and analyze reviews',
        replace_existing=True
    )

    scheduler.start()
    logger.info(f"Scheduler started. Running every {interval_minutes} minutes.")

    # Keep the script running
    try:
        asyncio.get_event_loop().run_forever()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
    except Exception as e:
        logger.error(f"Unexpected error in scheduler: {str(e)}", exc_info=True)
    finally:
        scheduler.shutdown()
        logger.info("Scheduler shut down.")

if __name__ == "__main__":
    start_scheduler()