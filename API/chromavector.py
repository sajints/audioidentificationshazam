from chromadb import PersistentClient
import uuid # For generating unique IDs
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# def get_audio_embedding(file_path):
#     audio, sr = preprocess_audio(file_path)
#     inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
#     with torch.no_grad():
#         embeddings = model(**inputs).last_hidden_state.mean(dim=1)
#     return embeddings.squeeze().numpy()
# Module-level variables



class chromavectordb:
    def __init__(self):
        self.client = PersistentClient(path="./audiodb")
        self.collection = self.client.get_or_create_collection(name="audio_fingerprints")

    def add(self, filename, query_fingerprint):
        """
        Saves one fingerprint and its file path to the database.
        """
        # Convert tuples to vectors of offsets
        fingerprint = [float(offset) for (hash_val, offset) in query_fingerprint]
        if len(fingerprint) != 150:  # Replace with your expected dimension
            raise ValueError(f"Embedding dimension mismatch: expected 150, got {len(fingerprint)}")
  
        self.collection.add(
            # We use the file path as the unique ID
            ids=[filename],            
            # We convert your numpy fingerprint to a plain list
            embeddings=[fingerprint],             
            # We store the path again in metadata so it's easy to read back
            metadatas=[{"path": filename}]
        )

    def addmultiple(self, file_paths, fingerprints):
        """
        file_paths: list of strings (e.g., ['path/to/audio.mp3'])
        fingerprints: list of numpy arrays or lists
        """
        # Convert fingerprints to list of lists if they are numpy arrays
        embeddings = [f.tolist() if hasattr(f, "tolist") else f for f in fingerprints]
        
        # Generate unique IDs for each file
        ids = [str(uuid.uuid4()) for _ in file_paths]
        
        # We store file paths in metadata so we can retrieve them later
        metadatas = [{"file_path": path} for path in file_paths]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(self, fingerprints, n_results=5):
        """
        query_fingerprint: a single numpy array or list
        """

        # Convert tuples to vectors of offsets
        query_fingerprint = [float(offset) for (hash_val, offset) in fingerprints]
        # Ensure it's a list and wrapped in another list (Chroma expects a list of queries)
        query_emb = query_fingerprint.tolist() if hasattr(query_fingerprint, "tolist") else query_fingerprint
        
        result = self.collection.query(
            query_embeddings=[query_emb],
            n_results=n_results
        )
        return result