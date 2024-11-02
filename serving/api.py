from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Modèle de données pour le mot
class WordRequest(BaseModel):
    word: str

@app.post("/duplicate")
def duplicate_word(request: WordRequest):
    duplicated_word = request.word * 2  # Duplique le mot
    return {"duplicated_word": duplicated_word}
