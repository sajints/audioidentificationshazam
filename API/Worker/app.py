import os
import time
from google.cloud import storage
from callapi import process_bucket
# Import your "Best-in-Class" logic from earlier
# from matcher import find_audio_match_robust_v2 

class AudioWorker:
    def __init__(self, bucket_name, local_download_path="./temp_audio"):
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        self.local_path = local_download_path
        os.makedirs(self.local_path, exist_ok=True)

    def download_blob(self, blob_name):
        """Downloads a file from GCP to local disk for processing."""
        local_file = os.path.join(self.local_path, blob_name)
        blob = self.bucket.blob(blob_name)
        blob.download_to_filename(local_file)
        return local_file

    def run_processing_loop(self):
        print("Worker started. Monitoring bucket...")
        # Get list of all files in bucket
        blobs = self.storage_client.list_blobs(self.bucket.name)

        for blob in blobs:
            if blob.name.endswith(('.mp3', '.wav')):
                print(f"Processing: {blob.name}")
                
                # 1. Fetch
                temp_file = self.download_blob(blob.name)
                
                # 2. Match (Separation of Concern: Logic happens HERE, not in API)
                try:
                    # result = find_audio_match_robust_v2("long_audio_path.mp3", temp_file)
                    # print(f"Match Result: {result}")
                    
                    # 3. Save Result (e.g., to BigQuery, Firestore, or SQL)
                    self.save_result_to_db(blob.name, {"status": "matched"})
                
                finally:
                    # 4. Cleanup
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

    def save_result_to_db(self, filename, result):
        """Placeholder for saving results to a database."""
        print(f"Saving results for {filename} to Database...")

if __name__ == "__main__":
    # In GKE, these would be Environment Variables
    worker = AudioWorker(bucket_name="your-audio-files-bucket")
    # worker.run_processing_loop()
    worker.process_bucket()