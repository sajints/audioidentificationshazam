from fingerprintsv3 import compare_fingerprints_v3, process_audio_v3
from database import store_fingerprints, find_match_in_db
import shutil
import os
from inspect import getsourcefile
from os.path import abspath
import json
import uuid
from robustmatch import find_audio_match_robust, find_audio_match_robust_v2


def _safe_upload_path(temp_dir: str, upload, fallback_prefix: str = "upload") -> str:
    name = getattr(upload, "filename", None)
    if not name or not str(name).strip():
        name = f"{fallback_prefix}_{id(upload)}"
    name = os.path.basename(str(name))
    return os.path.join(temp_dir, name)


def searchaudioservice(file1):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    path1 = _safe_upload_path(temp_dir, file1, "search")
    #path2 = os.path.join(temp_dir, file2.filename)

    # Save both files correctly
    with open(path1, "wb") as f:
        shutil.copyfileobj(file1.file, f)

    # 3. Generate Speech-Optimized Fingerprints
    fingerprints = process_audio_v3(path1)
    # print(f"fingerprints={fingerprints}")

    basepath = os.path.abspath(os.getcwd())
    savedpath = os.path.join(basepath,"SavedFiles") #GCP storage here
    matches = find_match_in_db(fingerprints)
    # data = json.load(matches)
    match_score = float(0)
    resultdict: dict[str, float] = {}
    for row in matches:
        # 2. Check if file actually exists before processing
        path2 = os.path.join(savedpath,row['filename'])
        print(f"current file={path2}")
        if not os.path.exists(path2):
            print(f"Warning: {path2} not found on disk.")
            continue
        
        # with open(path2, "wb") as f:
        fingerprint2 = process_audio_v3(path2)
        print(f"path1={path1} -- path2={path2}")
        match_score = find_audio_match_robust(path1,path2) #TBD
        # match_score = find_audio_match_robust_v2(path1,path2)
        resultdict[os.path.basename(path2)] = match_score

    #filename = file1.file
    # Process the entire list
    # hash,offset = [convert_fingerprint(h, t) for h, t in fingerprints]
    #store_fingerprints(counter,fingerprints,filename)
    
    # print(f"fingerprints={fingerprints}--counter={counter}--filename={filename}")
    return resultdict #find_match_in_db(fingerprints)

def saveaudioservice(file1):
    counter = uuid.uuid4()
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    path1 = _safe_upload_path(temp_dir, file1, "save")
    #path2 = os.path.join(temp_dir, file2.filename)

    # Save both files correctly
    with open(path1, "wb") as f:
        shutil.copyfileobj(file1.file, f)

    # 3. Generate Speech-Optimized Fingerprints
    fingerprints = process_audio_v3(path1)
    print(f"fingerprints={fingerprints}")

    filename = os.path.basename(path1)
    # Process the entire list
    # hash,offset = [convert_fingerprint(h, t) for h, t in fingerprints]
    store_fingerprints(counter,fingerprints,filename)
    print(f"fingerprints={fingerprints}--counter={counter}--filename={filename}")
    return "Saved the file"
