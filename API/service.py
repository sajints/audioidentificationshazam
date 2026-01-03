from fingerprintsv3 import compare_fingerprints_v3, process_audio_v3
from database import store_fingerprints, find_match_in_db
import shutil
import os
from inspect import getsourcefile
from os.path import abspath
import json
from robustmatch import find_audio_match_robust

counter = 2
def searchaudioservice(file1):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    path1 = os.path.join(temp_dir, file1.filename)
    #path2 = os.path.join(temp_dir, file2.filename)

    # Save both files correctly
    with open(path1, "wb") as f:
        shutil.copyfileobj(file1.file, f)

    # 3. Generate Speech-Optimized Fingerprints
    fingerprints = process_audio_v3(path1)
    print(f"fingerprints={fingerprints}")

    basepath = os.path.abspath(os.getcwd())
    savedpath = os.path.join(basepath,"SavedFiles")
    matches = find_match_in_db(fingerprints)
    # data = json.load(matches)
    match_score = float(0)
    resultdict: dict[str, float] = {}
    for row in matches:
        # 2. Check if file actually exists before processing
        filename = os.path.join(savedpath,row['filename'])
        print(f"current file={filename}")
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found on disk.")
            continue
        
        # with open(filename, "wb") as f:
        fingerprint2 = process_audio_v3(filename)
        match_score = find_audio_match_robust(path1,filename)
        resultdict[os.path.basename(filename)] = match_score

    #filename = file1.file
    # Process the entire list
    # hash,offset = [convert_fingerprint(h, t) for h, t in fingerprints]
    #store_fingerprints(counter,fingerprints,filename)
    
    # print(f"fingerprints={fingerprints}--counter={counter}--filename={filename}")
    return resultdict #find_match_in_db(fingerprints)

def saveaudioservice(file1):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    path1 = os.path.join(temp_dir, file1.filename)
    #path2 = os.path.join(temp_dir, file2.filename)

    # Save both files correctly
    with open(path1, "wb") as f:
        shutil.copyfileobj(file1.file, f)

    # 3. Generate Speech-Optimized Fingerprints
    fingerprints = process_audio_v3(path1)
    print(f"fingerprints={fingerprints}")

    filename = file1.filename
    # Process the entire list
    # hash,offset = [convert_fingerprint(h, t) for h, t in fingerprints]
    store_fingerprints(counter,fingerprints,filename)
    print(f"fingerprints={fingerprints}--counter={counter}--filename={filename}")
