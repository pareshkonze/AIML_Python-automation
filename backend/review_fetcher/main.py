import json
import asyncio
from common.database import init_db_pool, get_locations
from common.utils import setup_logging
from .google_api import process_google_reviews
from .facebook_api import process_facebook_reviews

logger = setup_logging()

async def handle_review_fetch(event, context):
    try:
        await init_db_pool()
        if event.get('source') == 'google':
            locations = await get_locations()
            results = []
            for location in locations:
                try:
                    result = await process_google_reviews(location)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing reviews for location {location.location_id}: {str(e)}", exc_info=True)
            return {
                'statusCode': 200,
                'body': json.dumps({"message": "Google reviews processed successfully", "results": results})
            }
        elif event.get('source') == 'facebook':
            try:
                await process_facebook_reviews("ptetutorials", "pte")
                await process_facebook_reviews("ieltstutorials.online", "ielts")
                await process_facebook_reviews("ccltutorials", "ccl")
                return {
                    'statusCode': 200,
                    'body': json.dumps({"message": "Facebook reviews processed successfully"})
                }
            except Exception as e:
                logger.error(f"Error processing Facebook reviews: {str(e)}", exc_info=True)
                return {
                    'statusCode': 500,
                    'body': json.dumps({"error": f"Error processing Facebook reviews: {str(e)}"})
                }
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({"error": "Invalid source specified"})
            }
    except Exception as e:
        logger.error(f"Error in handle_review_fetch: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }

if __name__ == "__main__":
    # For local testing
    asyncio.run(handle_review_fetch({'source': 'google'}, None))
    asyncio.run(handle_review_fetch({'source': 'facebook'}, None))