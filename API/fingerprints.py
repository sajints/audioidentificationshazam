import librosa
import numpy as np
import hashlib
from scipy import signal
from scipy.ndimage import maximum_filter

from collections import Counter, defaultdict
from typing import List, Tuple, Union, Any, Dict
import bisect

def detect_peaksnew(Sxx_db, threshold=-40, neighborhood_size=(20, 20)):
    # Create a mask of points above threshold
    mask = Sxx_db > threshold
    
    # Apply local maximum filter
    local_max = maximum_filter(Sxx_db, size=neighborhood_size) == Sxx_db #Calculate a multidimensional maximum filter.
    
    # Combine mask & local maxima
    peaks_mask = mask & local_max
    
    # Get coordinates (freq_idx, time_idx)
    peak_coords = np.argwhere(peaks_mask)
    
    # Return as list of (time, freq)
    return [(t, f) for f, t in peak_coords]

# def detect_peaksold(Sxx_db, threshold=-40):
#     peak_coords = []
#     for freq_idx in range(Sxx_db.shape[0]):
#         row = Sxx_db[freq_idx, :]
#         if np.max(row) < threshold:
#             continue
#         peaks = signal.find_peaks_cwt(row, np.arange(1, 10))
#         for time_idx in peaks:
#             peak_coords.append((time_idx, freq_idx))
#     return np.array(peak_coords)


def generate_fingerprints(peaks, fan_value=5):
    fingerprints = []
    #print(f"peaks - length={len(peaks)} -- peaks={peaks}")
    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if i + j < len(peaks):
                freq1 = int(peaks[i][1])
                freq2 = int(peaks[i + j][1])
                t1 = int(peaks[i][0])
                t2 = int(peaks[i + j][0])
                delta_t = t2 - t1
                if 0 <= delta_t <= 200:
                    hash_input = f"{freq1}|{freq2}|{delta_t}"
                    h = hashlib.sha1(hash_input.encode()).hexdigest()[0:20]
                    fingerprints.append((h, t1))
    #print(f"fingerprints - length={len(fingerprints)} -- fingerprints={fingerprints}")
    
    return fingerprints
# def generate_fingerprints(peaks, fan_value=5):
#     fingerprints = []
#     for i in range(len(peaks)):
#         for j in range(1, fan_value):
#             if i + j < len(peaks):
#                 freq1 = peaks[i][1]
#                 freq2 = peaks[i + j][1]
#                 t1 = peaks[i][0]
#                 t2 = peaks[i + j][0]
#                 delta_t = t2 - t1
#                 if delta_t >= 0 and delta_t <= 200:
#                     hash_input = f"{freq1}|{freq2}|{delta_t}"
#                     h = hashlib.sha1(hash_input.encode()).hexdigest()[0:20]
#                     fingerprints.append((h, t1))
#     return fingerprints


def process_audio(audio_path):
    y, sr = librosa.load(audio_path, duration=7, sr=44100) #sr = sampling rate. Load an audio file as a floating point time series.
    Sxx = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024)) # Short-time Fourier transform (STFT).
    Sxx_db = librosa.amplitude_to_db(Sxx, ref=np.max) # Convert an amplitude spectrogram to dB-scaled spectrogram.
    peaks = detect_peaksnew(Sxx_db)
    fingerprints = generate_fingerprints(peaks)
    return fingerprints



HashOrTuple = Union[str, int, Tuple[Union[str,int], float]]

def _normalize_hash(h: Any):
    """Try to convert hex-strings to ints, otherwise leave as-is."""
    if isinstance(h, int):
        return h
    if isinstance(h, str):
        try:
            return int(h, 16)
        except ValueError:
            try:
                return int(h)
            except ValueError:
                return h
    # If it's a tuple (hash, time), caller will unpack
    return h

def _is_time_tuple(x):
    return (isinstance(x, (list, tuple)) and len(x) >= 2 and
            (isinstance(x[1], (int, float))))

def compare_fingerprints(fp1: List[HashOrTuple],
                         fp2: List[HashOrTuple],
                         method: str = "auto",
                         threshold: float = 0.5,
                         time_tolerance: float = 2.0,
                         debug: bool = False):
    """
    Compare two fingerprint sequences.

    Args:
        fp1, fp2: lists of fingerprints. Supported formats:
            - list of hashes (hex-string like 'ab12' or ints)
            - list of (hash, time) tuples where time is seconds (or frames)
        method: 'auto' | 'jaccard' | 'multiset' | 'time'
            - 'auto': pick 'time' if timestamps detected, else 'multiset'
        threshold: threshold on returned score to declare a match
        time_tolerance: seconds (or same units as timestamps) for time-aligned matching
        debug: if True, also return a details dict with stats

    Returns:
        (score: float, is_match: bool) or (score, is_match, details) if debug=True
    """

    # Basic checks
    if not fp1 or not fp2:
        details = {"len1": len(fp1), "len2": len(fp2)}
        if debug:
            return 0.0, False, details
        return 0.0, False

    # Auto-detect timestamped fingerprints
    contain_time = _is_time_tuple(fp1[0]) or _is_time_tuple(fp2[0])
    if method == "auto":
        method = "time" if contain_time else "multiset"

    #print(f"Inside {method} method")
    # ----- TIME-ALIGNED method -----
    if method == "time":
        # Expect list of (hash, time)
        
        def build_map(fp):
            d = defaultdict(list)  # hash -> sorted list of times
            for item in fp:
                if _is_time_tuple(item):
                    h, t = item[0], float(item[1])
                else:
                    # If no time present, treat index as time
                    h, t = item, None
                h = _normalize_hash(h)
                if t is None:
                    # approximate time by incremental index (so matching still possible)
                    # use length of list as fallback
                    t = 0.0
                d[h].append(t)
            # sort times for each hash
            for k in d:
                d[k].sort()
            return d

        map1 = build_map(fp1)
        map2 = build_map(fp2)

        matched = 0
        total1 = sum(len(v) for v in map1.values())
        total2 = sum(len(v) for v in map2.values())

        # For each hash present in both, match times allowing tolerance
        for h in (set(map1.keys()) & set(map2.keys())):
            times1 = map1[h]
            times2 = map2[h]
            used = [False] * len(times2)
            for t1 in times1:
                # binary search candidate window in times2
                i = bisect.bisect_left(times2, t1 - time_tolerance)
                # scan forward while within tolerance
                while i < len(times2) and times2[i] <= t1 + time_tolerance:
                    if not used[i]:
                        matched += 1
                        used[i] = True
                        break
                    i += 1

        denom = min(total1, total2) if min(total1, total2) > 0 else max(total1, total2)
        score = matched / denom if denom > 0 else 0.0
        is_match = score >= threshold

        details = {
            "method": "time",
            "matched": matched,
            "total1": total1,
            "total2": total2,
            "denominator": denom
        }
        if debug:
            return score, is_match, details
        return score, is_match

    # ----- MULTISET (Counter overlap) -----
    # if method == "multiset":
    #     norm1 = [_normalize_hash(x if not _is_time_tuple(x) else x[0]) for x in fp1]
    #     norm2 = [_normalize_hash(x if not _is_time_tuple(x) else x[0]) for x in fp2]
    #     c1 = Counter(norm1)
    #     c2 = Counter(norm2)
    #     common = sum(min(c1[k], c2[k]) for k in (set(c1.keys()) & set(c2.keys())))
    #     # Normalize by min length so short clip inside long clip scores high
    #     denom = min(sum(c1.values()), sum(c2.values()))
    #     score = common / denom if denom > 0 else 0.0
    #     is_match = score >= threshold
    #     details = {
    #         "method": "multiset",
    #         "common_count": common,
    #         "len1": sum(c1.values()),
    #         "len2": sum(c2.values()),
    #         "unique1": len(c1),
    #         "unique2": len(c2),
    #         "denominator": denom
    #     }
    #     if debug:
    #         return score, is_match, details
    #     return score, is_match

    # # ----- JACCARD (set intersection / union) -----
    # if method == "jaccard":
    #     s1 = set(_normalize_hash(x if not _is_time_tuple(x) else x[0]) for x in fp1)
    #     s2 = set(_normalize_hash(x if not _is_time_tuple(x) else x[0]) for x in fp2)
    #     inter = len(s1 & s2)
    #     union = len(s1 | s2)
    #     score = inter / union if union > 0 else 0.0
    #     is_match = score >= threshold
    #     details = {"method": "jaccard", "intersection": inter, "union": union}
    #     if debug:
    #         return score, is_match, details
    #     return score, is_match

    
    return score, score >= threshold
        # Fallback
    if debug:
        return 0.0, False, {"error": "unknown method"}
    return 0.0, False