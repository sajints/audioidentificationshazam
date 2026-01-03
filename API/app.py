from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import sys
from fingerprints import process_audio, compare_fingerprints
from fingerprintsv2 import compare_fingerprints_robust, compare_fingerprints_v2, process_audio_v2

from sqllite import create_database, store_fingerprints, match_fingerprints
from chromavector import chromavectordb
 #convert_fingerprint, 
from service import searchaudioservice, saveaudioservice

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


@app.post("/saveaudio")
async def saveaudio(file1: UploadFile):
    return saveaudioservice(file1)

    # db = chromavectordb()
    # result = db.search(fingerprints,5)
    # print(f"result from search={result}")
    # if not result['ids'][0]: # Check if the result is empty
    #     saveresult=db.add(path1, fingerprints)
    #     print(f"result from save={saveresult}")

@app.post("/searchaudio")
async def searchaudio(file1: UploadFile):
    return searchaudioservice(file1)

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
    # fingerprints1 = process_audio(path1)
    # fingerprints2 = process_audio(path2)
#new code start
    # 3. Generate Speech-Optimized Fingerprints
    fingerprints1 = process_audio_v3(path1)
    fingerprints2 = process_audio_v3(path2)
#new code end
    with open("example.txt", "w", encoding="utf-8") as f:
        f.write(f"File1-{fingerprints1}\n\n\n")
        f.write(f"File2-{fingerprints2}\n")
    # Compare fingerprints
    #score, is_match = compare_fingerprints(fingerprints1, fingerprints2)
    # score, is_match = compare_fingerprints_robust (fingerprints1, fingerprints2)
    score, is_match = compare_fingerprints_v3 (fingerprints1, fingerprints2)

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