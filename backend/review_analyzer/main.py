import json
import asyncio
from common.database import init_db_pool, close_db_pool, execute_query
from common.utils import setup_logging
from common.models import ReviewInput
from .nlp_functions import analyze_single_review

logger = setup_logging()

async def handle_review_analysis(event, context):
    try:
        await init_db_pool()
        reviews = await get_unanalyzed_reviews(10)
        results = []
        for review in reviews:
            result = await analyze_single_review(review)
            if result:
                await mark_review_as_analyzed(review.review_id)
                results.append(result)
        
        return {
            'statusCode': 200,
            'body': json.dumps({"message": "Reviews analyzed successfully", "results": results})
        }
    except Exception as e:
        logger.error(f"Error in handle_review_analysis: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }
    finally:
        await close_db_pool()

async def get_unanalyzed_reviews(limit: int = 10):
    query = """
        SELECT r.review_id, r.review_text, r.source, gr.star_rating, COALESCE(gr.create_time, r.created_at) as review_date
        FROM reviews r
        LEFT JOIN google_review gr ON r.review_id = gr.review_id
        WHERE r.processed = false AND r.review_text != ''
        LIMIT $1
    """
    rows = await execute_query(query, limit)
    return [ReviewInput(
        review_id=row['review_id'],
        text=row['review_text'],
        source=row['source'],
        rating=str(row['star_rating'] or '0'),
        date=row['review_date']
    ) for row in rows]

async def mark_review_as_analyzed(review_id: str):
    query = "UPDATE reviews SET processed = true WHERE review_id = $1"
    await execute_query(query, review_id)
    logger.info(f"Marked review {review_id} as analyzed")

if __name__ == "__main__":
    asyncio.run(handle_review_analysis(None, None))