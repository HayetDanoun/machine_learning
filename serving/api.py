from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    value: float

@app.post("/predict")
async def predict(item: Item):
    return {
        "message": f"Received item with name: {item.name} and value: {item.value}"
    }
