from pydantic import BaseModel
from typing import Optional, Union

class BasicResponse(BaseModel):
    status: int
    message: str
    data: Optional[Union[dict, list]] = None


class BasicRequest(BaseModel):
    search_type: str = "text2text"


class SearchRequest(BaseModel):
    search_type: str = "text2text"
    query_text: Optional[str] = None
    query_image: Optional[str] = None
    top_k: int = 10
