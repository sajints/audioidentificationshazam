from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from fingerprints import process_audio, compare_fingerprints
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

# @app.post("/matchshazam")
# async def match_audio(file1: UploadFile = File(...), file2: UploadFile = File(...)):
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#     path1 = os.path.join(UPLOAD_FOLDER, file1.filename)
#     file1.file.seek(0)
#     with open(path1, "wb") as f:
#         shutil.copyfileobj(file1.file, f)

#     path2 = os.path.join(UPLOAD_FOLDER, file2.filename)
#     file2.file.seek(0)
#     with open(path2, "wb") as f:
#         shutil.copyfileobj(file2.file, f)

#     fingerprints1 = process_audio(path1)
#     fingerprints2 = process_audio(path2)

#     similarity, is_match = compare_fingerprints(fingerprints1, fingerprints2)
#     return {"match": is_match, "score": similarity}

@app.post("/matchshazam")
async def match_audio(file1: UploadFile, file2: UploadFile):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    path1 = os.path.join(temp_dir, file1.filename)
    path2 = os.path.join(temp_dir, file2.filename)

    # Save both files correctly
    with open(path1, "wb") as f:
        shutil.copyfileobj(file1.file, f)
    with open(path2, "wb") as f:
        shutil.copyfileobj(file2.file, f)

    # Extract fingerprints
    fingerprints1 = process_audio(path1)
    fingerprints2 = process_audio(path2)

    # Compare fingerprints
    score, is_match = compare_fingerprints(fingerprints1, fingerprints2)

    return {
        "similarity_score": score,
        "is_match": is_match
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",  # <filename>:<FastAPI instance>
        host="127.0.0.1",
        port=8005,
        reload=True
    )