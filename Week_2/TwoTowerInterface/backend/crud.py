from sqlalchemy.orm import Session
import models, schemas

# Function to create a new party
def register_search(db: Session, search: schemas.SearchCreate):
    db_query = models.Search(query=search.query,        
                            positive_passages=search.positive_passages,
                            negative_passages=search.negative_passages)
    db.add(db_query)
    db.commit()
    db.refresh(db_query)
    return db_query

