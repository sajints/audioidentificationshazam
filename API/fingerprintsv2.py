import librosa
import numpy as np
import hashlib
from scipy import signal
from scipy.ndimage import maximum_filter

from collections import Counter, defaultdict
from typing import List, Tuple, Union, Any, Dict
import bisect

# def create_speech_fingerprint(y: np.ndarray, sr: int) -> list:
#     """
#     Fingerprint optimized for speech with stable hashing and feature binning.
#     """
#     if len(y) < sr * 0.5:
#         return []

#     f_min = 300
#     f_max = 3000

#     # 1. Extract MFCCs
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, fmin=f_min, fmax=f_max)
    
#     # 2. Extract Spectral Centroid using Spectrogram
#     S, _ = librosa.magphase(librosa.stft(y=y))
#     centroid = librosa.feature.spectral_centroid(S=S, sr=sr)

#     hop_length = 512
#     frames_per_sec = sr / hop_length
#     fingerprints = []

#     # Process in 1-second chunks
#     for i in range(0, mfccs.shape[1], int(frames_per_sec)):
#         chunk_mfcc = mfccs[:, i : i + int(frames_per_sec)]
#         chunk_centroid = centroid[:, i : i + int(frames_per_sec)]
        
#         if chunk_mfcc.shape[1] == 0: 
#             continue
        
#         # Combine MFCC and Centroid into one vector
#         feature_vector = np.concatenate([
#             np.mean(chunk_mfcc, axis=1),
#             np.mean(chunk_centroid, axis=1)
#         ])
        
#         # --- THE FIX: QUANTIZATION ---
#         # Instead of rounding to 1 decimal, we "bin" the values.
#         # This means values like 14.1 and 14.8 both become 15.0.
#         # This is vital for different mics to produce the same hash.
#         fuzziness = 20.0
#         binned_features = np.round(feature_vector / fuzziness) * fuzziness
        
#         # --- THE FIX: STABLE HASHING ---
#         # Convert binned array to a string, then to a deterministic MD5 hash
#         feature_str = str(binned_features.tolist()).encode('utf-8')
#         stable_hash = hashlib.md5(feature_str).hexdigest()
        
#         timestamp = i / frames_per_sec
#         fingerprints.append((stable_hash, timestamp))
        
#     return fingerprints

def create_speech_fingerprint(y: np.ndarray, sr: int) -> list:
    if len(y) < sr * 0.5:
        return []

    # 1. Get the Spectrogram
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    
    # 2. Focus on Speech range (approx frames for 300Hz-3000Hz)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    speech_indices = np.where((freqs >= 300) & (freqs <= 3000))[0]
    S_speech = S[speech_indices, :]

    fingerprints = []
    hop_length = 512
    frames_per_sec = int(sr / hop_length)

    # 3. For every second, find the 'Landmarks' (Top frequencies)
    for t in range(0, S_speech.shape[1], frames_per_sec):
        chunk = S_speech[:, t : t + frames_per_sec]
        if chunk.shape[1] == 0: continue

        # Find the indices of the top 3 loudest frequency bins in this second
        # We sum across the second to find the most persistent frequencies
        persistence = np.sum(chunk, axis=1)
        top_bins = np.argsort(persistence)[-3:] # Get indices of top 3 peaks
        top_bins = np.sort(top_bins) # Sort them to keep the hash consistent
        
        # Create a simple, stable string hash from these 3 peaks
        # Example: "bin14_bin25_bin60"
        h = "_".join([f"b{b}" for b in top_bins])
        
        timestamp = t / frames_per_sec
        fingerprints.append((h, timestamp))
        
    return fingerprints

def process_audio_v2(audio_path):
    # Standardizing sample rate is key for matching
    y1, sr1 = librosa.load(audio_path, sr=22050, mono=True)
    return create_speech_fingerprint(y1, sr1)


def compare_fingerprints_robust(fp1, fp2, threshold=0.5, time_tolerance=0.5, debug=False):
    """
    Robust comparison that handles offsets and different silence durations.
    Uses a histogram of time offsets to find the true alignment.
    """
    if not fp1 or not fp2:
        return 0.0, False

    # 1. Map hashes to their timestamps
    def get_map(fp):
        d = defaultdict(list)
        for h, t in fp:
            d[h].append(float(t))
        return d

    map1 = get_map(fp1)
    map2 = get_map(fp2)

    # 2. Find all possible time offsets between matching hashes
    # If the files are the same, the correct offset will appear many times
    offsets = []
    common_hashes = set(map1.keys()) & set(map2.keys())
    
    for h in common_hashes:
        for t1 in map1[h]:
            for t2 in map2[h]:
                offsets.append(round(t1 - t2, 1)) # Round to 0.1s for grouping

    if not offsets:
        return 0.0, False

    # 3. Find the most frequent offset (The 'True' alignment)
    offset_counts = Counter(offsets)
    best_offset, hit_count = offset_counts.most_common(1)[0]

    # 4. Calculate score based on how many hashes align at that specific offset
    # We allow a small tolerance around the best_offset
    total_potential = min(len(fp1), len(fp2))
    
    # Refined count: how many hits are within the tolerance of our best offset
    true_matches = 0
    for offset, count in offset_counts.items():
        if abs(offset - best_offset) <= time_tolerance:
            true_matches += count

    score = true_matches / total_potential if total_potential > 0 else 0.0
    is_match = score >= threshold

    if debug:
        return score, is_match, {
            "best_offset_seconds": best_offset,
            "hits_at_offset": hit_count,
            "total_matches_in_window": true_matches,
            "total_hashes": total_potential
        }
    return score, is_match
    

def compare_fingerprints_v2(fp1, fp2, threshold=0.3, time_tolerance=2.0):
    """
    v2: Optimized for speech-heavy meetings with high noise/gain differences.
    """
    if not fp1 or not fp2: return 0.0, False
    # print(f"start fp1={fp1}")
    # print(f"start fp2={fp2}")

    map1 = defaultdict(list)
    for h, t in fp1: map1[h].append(t)
    
    map2 = defaultdict(list)
    for h, t in fp2: map2[h].append(t)

    # print(f"map1.keys()={map1.keys()}")
    # print(f"map2.keys()={map2.keys()}")

    # Calculate all possible time differences
    offsets = []
    for h in set(map1.keys()) & set(map2.keys()):
        for t1 in map1[h]:
            for t2 in map2[h]:
                offsets.append(round(t1 - t2, 1))
                # print(f"hash={h}")

    # print(f"offsets={offsets}")

    if not offsets: return 0.0, False

    # Find the BEST offset
    counts = Counter(offsets)
    # Get the top 3 most common offsets (to handle slight jitter)
    top_offsets = [obj[0] for obj in counts.most_common(3)]
    
    # Check how many matches fall within 1 second of the BEST offset
    best_offset = top_offsets[0]
    matched_hashes = 0
    for h in set(map1.keys()) & set(map2.keys()):
        for t1 in map1[h]:
            for t2 in map2[h]:
                if abs((t1 - t2) - best_offset) <= time_tolerance:
                    matched_hashes += 1
                    break # Count this hash only once

    # Use a relative denominator: how many hashes COULD have matched?
    score = matched_hashes / min(len(fp1), len(fp2))
    is_match = score >= threshold
    print(f"score={score}")

    return score, is_match