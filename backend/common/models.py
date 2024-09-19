from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class Location(BaseModel):
    id: int
    location_id: str
    location_name: str
    website_name: str

    class Config:
        orm_mode = True

class GoogleReview(BaseModel):
    id: Optional[int] = None
    review_id: str
    reviewer_display_name: Optional[str] = None
    reviewer_profile_photo_url: Optional[str] = None
    star_rating: Optional[str] = None
    comment: Optional[str] = None
    create_time: datetime
    update_time: datetime
    review_reply_comment: Optional[str] = None
    name: Optional[str] = None
    deleted_date: Optional[datetime] = None
    is_deleted: bool = False
    review_type: Optional[str] = None
    website_name: Optional[str] = None
    is_reply: Optional[int] = None
    reply_date: Optional[datetime] = None
    review_reply_update_time: Optional[datetime] = None
    tbl_location_id: int
    location_id: Optional[str] = None
    location: Optional[str] = None

class FacebookReview(BaseModel):
    facebook_id: str
    website_name: str
    review_data: Dict[str, Any]

class ReviewInput(BaseModel):
    review_id: str
    text: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1, max_length=50)
    rating: str  # Keep this as a string to maintain compatibility
    date: Optional[datetime] = None