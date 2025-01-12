from pydantic import BaseModel
from typing import List

# Pydantic model for creating a new party
class SearchCreate(BaseModel):
    query: str
    positive_passages: List[int]
    negative_passages: List[int]

# Pydantic model for reading a party (response model)
class Search(SearchCreate):
    id: int

    class Config:
        from_attributes = True  # This allows Pydantic to work with SQLAlchemy models
