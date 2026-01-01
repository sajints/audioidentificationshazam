import librosa
import numpy as np
import hashlib
from scipy import signal
from scipy.ndimage import maximum_filter

from collections import Counter, defaultdict
from typing import List, Tuple, Union, Any, Dict
import bisect

def process_audio_v3(audio_path):
    # Standardizing sample rate is key for matching
    y1, sr1 = librosa.load(audio_path, sr=22050, mono=True)
    return create_speech_fingerprint(y1, sr1)

def create_speech_fingerprint(y: np.ndarray, sr: int) -> list:
    if len(y) < sr * 0.5:
        return []

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    speech_indices = np.where((freqs >= 300) & (freqs <= 3000))[0]
    S_speech = S[speech_indices, :]

    fingerprints = []
    hop_length = 512
    frames_per_sec = int(sr / hop_length)

    for t in range(0, S_speech.shape[1], frames_per_sec):
        chunk = S_speech[:, t : t + frames_per_sec]
        if chunk.shape[1] == 0: continue

        persistence = np.sum(chunk, axis=1)
        # Get top 3 peaks
        top_bins = np.argsort(persistence)[-3:]
        top_bins = np.sort(top_bins)
        
        # --- IMPROVEMENT: GENERATE MULTIPLE COMBINATIONS ---
        # Instead of one "b1_b2_b3", we create 3 pairs.
        # This way, if we miss one peak, the other pairs still match!
        b1, b2, b3 = top_bins
        combos = [f"{b1}_{b2}", f"{b2}_{b3}", f"{b1}_{b3}"]
        
        timestamp = t / frames_per_sec
        for c in combos:
            fingerprints.append((c, timestamp))
        
    return fingerprints

def compare_fingerprints_v3(fp1, fp2, threshold=0.3, time_tolerance=1.0):
    map1 = defaultdict(list)
    for h, t in fp1: map1[h].append(t)
    
    map2 = defaultdict(list)
    for h, t in fp2: map2[h].append(t)

    offsets = []
    common_hashes = set(map1.keys()) & set(map2.keys())
    
    for h in common_hashes:
        for t1 in map1[h]:
            for t2 in map2[h]:
                offsets.append(round(t1 - t2, 1))

    if not offsets: return 0.0, False

    counts = Counter(offsets)
    best_offset, _ = counts.most_common(1)[0]

    # Calculate matches at the best offset
    matched_timestamps = set()
    total_timestamps = set(t for h, t in fp1)

    for h in common_hashes:
        for t1 in map1[h]:
            for t2 in map2[h]:
                if abs((t1 - t2) - best_offset) <= time_tolerance:
                    matched_timestamps.add(t1)

    # Score is the % of the file duration that found a match
    score = len(matched_timestamps) / len(total_timestamps) if total_timestamps else 0.0
    
    # Optional: Boost the score if it's a clear winner
    # Meeting audio is noisy, so 0.7 is often the "real world" 1.0
    final_score = min(1.0, score * 1.2) 
    
    return final_score, final_score >= threshold