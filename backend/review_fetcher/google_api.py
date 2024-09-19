import os
import logging
import aiohttp
from datetime import datetime
import json
import asyncio
from common.database import execute_query, execute_procedure
from common.models import Location, GoogleReview
from common.utils import ensure_directory_exists, save_reviews_to_file

logger = logging.getLogger(__name__)

MAX_REQUESTS_PER_MINUTE = int(os.getenv('MAX_REQUESTS_PER_MINUTE', '30'))
request_times = []

async def rate_limited_api_call(func, *args, **kwargs):
    global request_times
    now = datetime.now()
    
    # Remove requests older than 1 minute
    request_times = [t for t in request_times if (now - t).total_seconds() < 60]
    
    if len(request_times) >= MAX_REQUESTS_PER_MINUTE:
        sleep_time = 60 - (now - request_times[0]).total_seconds()
        if sleep_time > 0:
            logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
            await asyncio.sleep(sleep_time)
    
    result = await func(*args, **kwargs)
    request_times.append(datetime.now())
    return result

async def get_access_token(refresh_token: str, app_key: str, secret_key: str):
    url = "https://oauth2.googleapis.com/token"
    data = {
        "client_id": app_key,
        "client_secret": secret_key,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token"
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, data=data) as response:
                response.raise_for_status()
                result = await response.json()
                return result.get("access_token")
        except aiohttp.ClientError as e:
            logger.error(f"Error getting access token: {e}", exc_info=True)
            raise

async def get_business_api_response(next_token: str, location: Location):
    app_key = os.getenv("GOOGLE_APP_KEY")
    secret_key = os.getenv("GOOGLE_SECRET_KEY")
    refresh_token = os.getenv(f"{location.website_name.upper()}_REFRESH_TOKEN")
    
    access_token = await get_access_token(refresh_token, app_key, secret_key)
    
    api_url = f"https://mybusiness.googleapis.com/v4/accounts/107458317490405300214/locations/{location.location_id}/reviews"
    if next_token:
        api_url += f"?pageToken={next_token}"
    
    headers = {"Authorization": f"Bearer {access_token}"}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(api_url, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching Google reviews: {e}", exc_info=True)
            raise

async def process_google_reviews(location: Location):
    next_token = ""
    last_create_time = datetime.now()
    all_reviews = []

    while True:
        api_response = await rate_limited_api_call(get_business_api_response, next_token, location)
        
        if not api_response:
            break

        for review in api_response.get('reviews', []):
            update_time = datetime.fromisoformat(review['updateTime'].rstrip('Z'))
            if update_time <= datetime(2022, 7, 1):
                last_create_time = datetime.fromisoformat(review['createTime'].rstrip('Z'))
                break

            existing_review = await get_existing_review(review['reviewId'])
            if not existing_review:
                google_review = GoogleReview(
                    review_id=review['reviewId'],
                    reviewer_display_name=review.get('reviewer', {}).get('displayName'),
                    reviewer_profile_photo_url=review.get('reviewer', {}).get('profilePhotoUrl'),
                    star_rating=review.get('starRating', 'UNSPECIFIED'),
                    comment=review.get('comment', ''),
                    create_time=datetime.fromisoformat(review['createTime'].rstrip('Z')),
                    update_time=update_time,
                    review_reply_comment=review.get('reviewReply', {}).get('comment'),
                    name=review.get('name'),
                    review_type='google-business',
                    website_name=location.website_name,
                    is_reply=0,
                    reply_date=None,
                    review_reply_update_time=review.get('reviewReply', {}).get('updateTime'),
                    tbl_location_id=int(location.id),
                    location_id=location.location_id,
                    location=location.location_name
                )
                await insert_google_review(google_review)

            all_reviews.append(review)

        if last_create_time <= datetime(2022, 7, 1) or 'nextPageToken' not in api_response:
            break

        next_token = api_response['nextPageToken']

    save_reviews_to_file(all_reviews, location.location_id)

    await delete_review_data(location)

    return json.dumps(all_reviews)

async def get_existing_review(review_id: str):
    query = """
        SELECT gr.*, l.location_id, l.location_name as location
        FROM google_review gr
        LEFT JOIN tbl_location l ON gr.tbl_location_id = l.id
        WHERE gr.review_id = $1
    """
    result = await execute_query(query, review_id)
    if result:
        return GoogleReview(**dict(result[0]))
    return None

async def insert_google_review(review: GoogleReview):
    await execute_procedure("insert_google_review",
        review.review_id,
        review.reviewer_display_name,
        review.reviewer_profile_photo_url,
        review.star_rating,
        review.comment,
        review.create_time,
        review.update_time,
        review.review_reply_comment,
        review.name,
        review.deleted_date,
        review.is_deleted,
        review.review_type,
        review.website_name,
        review.is_reply,
        review.reply_date,
        review.review_reply_comment,
        review.review_reply_update_time,
        review.location_id
    )
    logger.info(f"Inserted/Updated Google review: {review.review_id}")

async def delete_review_data(location: Location):
    try:
        file_path = f'temp_uploads/google_review_{location.location_id}.txt'
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return
        
        with open(file_path, 'r') as f:
            stored_reviews = json.load(f)
        
        db_reviews = await fetch_reviews_for_deletion(False, location.id, datetime(2022, 7, 1))
        
        for db_review_id in db_reviews:
            if not any(r['reviewId'] == db_review_id for r in stored_reviews):
                await update_google_review_as_deleted(db_review_id)
    
    except Exception as e:
        logger.error(f"Error in delete_review_data for location {location.location_id}: {e}", exc_info=True)

async def fetch_reviews_for_deletion(is_deleted: bool, tbl_location_id: int, create_time: datetime):
    query = """
        SELECT review_id FROM google_review 
        WHERE is_deleted = $1 AND tbl_location_id = $2 AND create_time >= $3
    """
    results = await execute_query(query, is_deleted, tbl_location_id, create_time)
    return [row['review_id'] for row in results]

async def update_google_review_as_deleted(review_id: str):
    query = """
        UPDATE google_review 
        SET is_deleted = true, deleted_date = CURRENT_TIMESTAMP 
        WHERE review_id = $1
    """
    await execute_query(query, review_id)
    logger.info(f"Updated review {review_id} as deleted")