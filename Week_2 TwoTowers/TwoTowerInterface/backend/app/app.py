from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import models, schemas, crud
from sqlalchemy.orm import Session
from database import engine, get_db
from twotowermodel import generate_passages
from .ml_models.ml_model import run_two_tower_inference
from .ml_models.update_model import update_model as model_update_function
from typing import Any, List

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define the request body model
class Query(BaseModel):
    query: str  # The expected input field

class InferenceResponse(BaseModel):
    result: Any  # Replace 'Any' with the appropriate type
    indexes: List[int]  # Adjust the type as needed


@app.post("/api/user_input", response_model=InferenceResponse)
async def pass_user_query(query: Query):
    result, indexes = run_two_tower_inference(query.query)
    print("hit me", result, indexes)
    return {"result": result, "indexes": indexes}

@app.post("/api/update_model")
async def update_model(search: schemas.SearchCreate,  background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    print("API caught the call", search.positive_passages)

    try:
        background_tasks.add_task(model_update_function, search.query, search.positive_passages, search.negative_passages)        
    except Exception as e:
        print("Error updating model:", e)
        raise HTTPException(status_code=500, detail="Failed to update the model")

    return crud.register_search(db=db, search=search)