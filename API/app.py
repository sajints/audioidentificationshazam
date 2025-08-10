from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from fingerprints import process_audio
from sqllite import create_database, store_fingerprints, match_fingerprints

app = FastAPI()
# Allow all origins (or restrict to specific ones)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

create_database()

UPLOAD_FOLDER = "audio_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/fingerprint")
async def fingerprint_audio(file: UploadFile = File(...), song_id: str = Form(...)):
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    fingerprints = process_audio(path)
    store_fingerprints(fingerprints, song_id=song_id)
    return {"message": "Fingerprinting completed", "fingerprints": len(fingerprints)}

@app.post("/match")
async def match_audio(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    fingerprints = process_audio(path)
    match = match_fingerprints(fingerprints)
    return {"match": match[0], "score": match[1]}
