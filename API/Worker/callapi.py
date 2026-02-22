import os
import requests
from google.cloud import storage

# --- CONFIGURATION ---
API_URL = "http://your-api-service:8000/saveaudio"  # The address of your FastAPI
BUCKET_NAME = "your-audio-bucket"
TEMP_DIR = "./worker_temp"

os.makedirs(TEMP_DIR, exist_ok=True)

def process_bucket():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = client.list_blobs(BUCKET_NAME)

    for blob in blobs:
        if blob.name.endswith(('.wav', '.mp3')):
            print(f"--- Processing: {blob.name} ---")
            
            # 1. Download from GCP to Worker disk
            local_file_path = os.path.join(TEMP_DIR, blob.name)
            blob.download_to_filename(local_file_path)
            
            # 2. CALL THE API
            # We open the file in 'rb' (read-binary) mode
            with open(local_file_path, 'rb') as f:
                files = {'file1': (blob.name, f, 'audio/mpeg')}
                try:
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        print(f"Success: API received {blob.name}. Response: {response.json()}")
                    else:
                        print(f"API Error: {response.status_code} - {response.text}")
                
                except Exception as e:
                    print(f"Connection Failed: {e}")

            # 3. Cleanup temp file
            os.remove(local_file_path)

if __name__ == "__main__":
    process_bucket()