from sqlalchemy import Column, Integer, String, ARRAY
from database import Base

class Search(Base):
    __tablename__ = "search"
    
    id = Column(Integer, primary_key=True)
    query = Column(String, index=True, nullable=False)
    positive_passages = Column(ARRAY(Integer), default=[])
    negative_passages = Column(ARRAY(Integer), default=[])
