import librosa
import numpy as np
import scipy.signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')
# from chroma import chromavectordb
import librosa
#import torch
from scipy.ndimage import maximum_filter
from scipy.spatial.distance import cosine
import math

def normalize_audio_signal(y: np.ndarray, method='robust') -> np.ndarray:
    """
    Enhanced normalization with multiple methods
    """
    if len(y) == 0:
        return y
    
    #print(f"before offset removal- y={y}")

    # Remove DC offset
    y = y - np.mean(y)
    #print(f"after offset removal- y={y}")
    if method == 'robust':
        # Use robust/MAD statistics (less sensitive to outliers/silence)
        mad = np.median(np.abs(y - np.median(y)))
        #print(f"Inside robust- mad={mad}")

        if mad < 1e-8:
            std_dev = np.std(y) #call to get standard deviation
            if std_dev < 1e-8:
                return np.zeros_like(y)
            #print(f"Inside robust- std_dev={std_dev}")
            
            return y / std_dev
         
        res = y / (1.4826 * mad)     # main formula for robust
        #print(f"Inside robust -- res={res}")  
        return res
    else:
        # Standard normalization
        std_dev = np.std(y) #call to get standard deviation
        #print(f"Inside Standard normalization- std_dev={std_dev}")
        
        if std_dev < 1e-8:
            print(f"Inside Standard normalization- 1e-8={1e-8} np.zeros_like(y)={np.zeros_like(y)}")

            return np.zeros_like(y)
        res = y / std_dev # main formula for Standard normalization
        print(f"Inside Standard normalization- std_dev={std_dev} --- res={res}")  
        return res

def detect_silence_segments(y: np.ndarray, sr: int, silence_threshold: float = 0.01, min_silence_duration: float = 0.5) -> list:
    """
    Detect silent segments in audio
    """
    # Calculate RMS energy in small windows
    frame_length = int(0.1 * sr)
    hop_length = frame_length // 4
    
    # Flatten the RMS result because librosa returns it as (1, n_frames)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Find silent frames
    silent_frames = rms < silence_threshold
    
    # Convert frame indices to time
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    # Group consecutive silent frames
    silent_segments = []
    start_time = None
    
    for i, is_silent in enumerate(silent_frames):
        if is_silent and start_time is None:
            start_time = times[i]
        elif not is_silent and start_time is not None:
            duration = times[i] - start_time
            if duration >= min_silence_duration:
                silent_segments.append((start_time, times[i]))
            start_time = None
    
    # Handle case where audio ends with silence
    if start_time is not None:
        duration = times[-1] - start_time
        if duration >= min_silence_duration:
            silent_segments.append((start_time, times[-1]))
    
    return silent_segments

def create_audio_fingerprint(y: np.ndarray, sr: int, method='mfcc_chroma') -> np.ndarray:
    """
    Create robust audio fingerprint using multiple features
    """
    if len(y) < sr * 0.1:
        return np.array([])
    
    features = []
    
    if 'mfcc' in method:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        features.append(np.mean(mfcc, axis=1))
    
    if 'chroma' in method:
    # Use chroma_stft (Standard) or chroma_cqt (More robust for music)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512) 
        features.append(np.mean(chroma, axis=1))
    
    if 'spectral' in method:
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)
        
        features.extend([
            np.mean(spectral_centroid),
            np.mean(spectral_rolloff), 
            np.mean(spectral_bandwidth)
        ])
    
    if not features:
        return np.array([])
    
    return np.concatenate([f.flatten() if f.ndim > 0 else [f] for f in features])

def sliding_window_correlation(long_signal: np.ndarray, short_signal: np.ndarray, window_size: int = None, step_size: int = None) -> tuple:
    """
    Perform sliding window correlation for better local matching
    """
    if window_size is None:
        window_size = len(short_signal)
    if step_size is None:
        step_size = window_size // 4
    
    if len(long_signal) < window_size:
        return 0.0, 0
    
    best_score = -1
    best_offset = 0
    
    for offset in range(0, len(long_signal) - window_size + 1, step_size):
        window = long_signal[offset:offset + window_size]
        
        if len(window) == len(short_signal):
            try:
                corr, _ = pearsonr(window, short_signal)
                if not np.isnan(corr) and corr > best_score:
                    best_score = corr
                    best_offset = offset
            except:
                continue
    
    return best_score, best_offset

def find_audio_match_robust(long_file_path: str, short_file_path: str, sr: int = 22050) -> dict:
#def find_audio_match_robust(y_long: np.ndarray, y_short: np.ndarray, sr: int = 22050) -> dict:

    """
    Enhanced audio matching with multiple detection strategies
    y: 
    """
    try:
        y_long, _ = librosa.load(long_file_path, sr=sr, mono=True)
        y_short, _ = librosa.load(short_file_path, sr=sr, mono=True)
    except Exception as e:
        return {"error": f"Failed to load audio files: {e}"}

    if len(y_short) > len(y_long):
        y_long, y_short = y_short, y_long
        files_swapped = True
    else:
        files_swapped = False

    if len(y_long) == len(y_short):
        y_long_norm = normalize_audio_signal(y_long, method='robust')
        y_short_norm = normalize_audio_signal(y_short, method='robust')
        # Returns True only if every element in y_long_norm and y_short_norm satisfies:
        # ∣∣y long_norm  − y short_norm  ∣≤(atol+rtol⋅∣y short_norm  ∣)
        if np.allclose(y_long_norm, y_short_norm, rtol=1e-3, atol=1e-6):
            return {
                "match_type": "EXACT_DUPLICATE",
                "match_score": 3.0,
                "offset_seconds": 0.0,
                "confidence": "VERY_HIGH",
                "conclusion": "Files are exact duplicates (possibly with different names)"
            }

    silence_long = detect_silence_segments(y_long, sr)
    silence_short = detect_silence_segments(y_short, sr)

    y_long_norm = normalize_audio_signal(y_long, method='robust')
    y_short_norm = normalize_audio_signal(y_short, method='robust')
    
    print(f"after normalize_audio_signal - y_long_norm={y_long_norm}")            
    print(f"after normalize_audio_signal - y_short_norm={y_short_norm}")            

    if np.sum(np.abs(y_short_norm)) == 0 or np.sum(np.abs(y_long_norm)) == 0:
        return {
            "match_type": "NO_MATCH",
            "match_score": 0.0,
            "offset_seconds": 0.0,
            "confidence": "HIGH",
            "conclusion": "One or both files contain only silence"
        }
    
    correlation = scipy.signal.correlate(y_long_norm, y_short_norm, mode='full')
    print(f"after scipy.signal.correlate - correlation={correlation}")            

    max_corr_idx = np.argmax(correlation)
    print(f"after np.argmax - max_corr_idx={max_corr_idx}")            

    max_score_corr = correlation[max_corr_idx] / len(y_short_norm)
    print(f"after max_score_corr={max_score_corr}")            

    offset_samples_corr = max_corr_idx - (len(y_short_norm) - 1)
    offset_seconds_corr = offset_samples_corr / sr
    print(f"after offset_samples_corr={offset_samples_corr}--offset_seconds_corr={offset_seconds_corr}")            

    window_score, window_offset = sliding_window_correlation(y_long_norm, y_short_norm)
    window_offset_seconds = window_offset / sr
    print(f"after window_score={window_score}--window_offset={window_offset}--window_offset_seconds={window_offset_seconds}")            

    fingerprint_long = create_audio_fingerprint(y_long, sr)
    fingerprint_short = create_audio_fingerprint(y_short, sr)
    
    #print(f"fingerprint_long={fingerprint_long}--fingerprint_short={fingerprint_short}")            

    feature_similarity = 0.0
    if len(fingerprint_long) > 0 and len(fingerprint_short) > 0:
        try:
            feature_similarity, _ = pearsonr(fingerprint_long, fingerprint_short)
            if np.isnan(feature_similarity):
                feature_similarity = 0.0
        except:
            feature_similarity = 0.0

    segment_matches = []
    segment_size = sr * 5
    
    for i in range(0, len(y_long) - segment_size, segment_size // 2):
        segment_long = y_long[i:i + segment_size]
        if len(segment_long) < segment_size:
            continue
            
        segment_long_norm = normalize_audio_signal(segment_long)
        
        for j in range(0, len(y_short) - segment_size, segment_size // 2):
            segment_short = y_short[j:j + segment_size]
            if len(segment_short) < segment_size:
                continue
                
            segment_short_norm = normalize_audio_signal(segment_short)
            
            try:
                seg_corr, _ = pearsonr(segment_long_norm, segment_short_norm)
                if not np.isnan(seg_corr) and seg_corr > 0.7:
                    segment_matches.append({
                        'score': seg_corr,
                        'long_start': i / sr,
                        'short_start': j / sr
                    })
            except:
                continue

    print(f"max_score_corr={max_score_corr}")            
    print(f"window_score={window_score}")            
    print(f"feature_similarity={feature_similarity}")            
    print(f"segment_matches={segment_matches}--len(segment_matches)={len(segment_matches)}")            

    scores = {
        'correlation': max(0, max_score_corr),
        'sliding_window': max(0, window_score),
        'feature_similarity': max(0, feature_similarity),
        'segment_matches': len(segment_matches)
    }
    
    # final_score = (
    #     scores['correlation'] * 0.4 +
    #     scores['sliding_window'] * 0.4 +
    #     scores['feature_similarity'] * 0.2
    # )
    final_score = (
        scores['correlation'] * 0.25 +
        scores['sliding_window'] * 0.25 +
        scores['feature_similarity'] * 0.50
    )
    print(f"final_score={final_score}")            
    # 1. Very Strong Match (Exponents of both texture and timing)
    if final_score >= 0.9 or scores['sliding_window'] >= 0.95:
        match_type = "EXACT_MATCH"
        confidence = "VERY_HIGH"
        conclusion = "The shorter audio is contained within the longer audio with high confidence"
        offset_seconds = offset_seconds_corr if scores['correlation'] > scores['sliding_window'] else window_offset_seconds
    # 2. Strong Identification (Relies more on the 'Texture'/Fingerprint)    
    elif final_score >= 0.7 or scores['segment_matches'] >= 3:
        match_type = "PARTIAL_MATCH"
        confidence = "HIGH"
        conclusion = f"Partial match detected with {scores['segment_matches']} matching segments"
        offset_seconds = window_offset_seconds
    # 3. Fuzzy/Noisy Match    
    elif final_score >= 0.5 or (scores['feature_similarity'] > 0.6 and scores['segment_matches'] >= 1):
        match_type = "WEAK_MATCH"
        confidence = "MEDIUM"
        conclusion = "Weak similarity detected - possible match with significant differences"
        offset_seconds = window_offset_seconds
        
    else:
        match_type = "NO_MATCH"
        confidence = "HIGH"
        conclusion = "No significant similarity found between the audio files"
        offset_seconds = 0.0

    # 1. Clean the scores dictionary
    cleaned_scores = {k: float(v) for k, v in scores.items()}

    # 2. Clean the segment matches (cast 'score' to float)
    cleaned_segments = []
    if segment_matches:
        for seg in segment_matches[:5]:
            cleaned_segments.append({
                'score': float(seg['score']),
                'long_start': float(seg['long_start']),
                'short_start': float(seg['short_start'])
            })

    # 3. Clean the silence segments (cast tuples of np.floats to standard floats)
    cleaned_silence = {
        "long_file": [[float(start), float(end)] for start, end in silence_long],
        "short_file": [[float(start), float(end)] for start, end in silence_short]
    }

    return {
        "match_type": match_type,
        "match_score": float(final_score),
        # "offset_seconds": float(offset_seconds),
        "confidence": confidence,
        "conclusion": conclusion,
        "detailed_scores": cleaned_scores,
        # "silence_segments": cleaned_silence,
        "segment_matches": scores['segment_matches'], #cleaned_segments,
        "Length of segment_matches": len(cleaned_segments), #len(scores['segment_matches']) # 
        # "files_swapped": files_swapped
    }

    # return {
    #     "match_type": match_type,
    #     "match_score": float(final_score),
    #     "offset_seconds": float(offset_seconds),
    #     "confidence": confidence,
    #     "conclusion": conclusion,
    #     "detailed_scores": scores,
    #     "silence_segments": {
    #         "long_file": silence_long,
    #         "short_file": silence_short
    #     },
    #     "segment_matches": segment_matches[:5] if segment_matches else [],
    #     "files_swapped": files_swapped
    # }

def compare_audio_files_batch(file_pairs: list, sr: int = 22050) -> list:
    """
    Compare multiple audio file pairs
    """
    results = []
    for long_path, short_path in file_pairs:
        result = find_audio_match_robust(long_path, short_path, sr)
        result['file_pair'] = (long_path, short_path)
        results.append(result)
    return results


def find_audio_match_robust_v2(long_file_path: str, short_file_path: str, sr: int = 22050) -> dict:
    """
    3-Tier Robust Audio Matcher:
    1. Spectral Landmark Fingerprinting (Time-invariant)
    2. FFT Cross-Correlation (Precise Offset)
    3. Neural Embedding Similarity (Semantic Match)
    """
    try:
        y_long, _ = librosa.load(long_file_path, sr=sr, mono=True)
        y_short, _ = librosa.load(short_file_path, sr=sr, mono=True)
    except Exception as e:
        return {"error": f"Load failed: {e}"}

    if len(y_short) > len(y_long):
        y_long, y_short = y_short, y_long
        files_swapped = True
    else:
        files_swapped = False
    # --- STRATEGY 1: Spectral Landmarks (The "Shazam" Constellation) ---
    def get_constellation(y):
        # Convert to Spectrogram
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        # Find local peaks (landmarks)
        peaks = maximum_filter(S, size=(20, 20)) == S
        # Threshold to ignore silence
        peaks &= (S > np.mean(S))
        return np.argwhere(peaks)

    peaks_long = get_constellation(y_long)
    peaks_short = get_constellation(y_short)
    
    # Calculate Jaccard similarity of peak distributions (Simplified Fingerprint)
    # In a production app, you'd hash pairs of peaks here
    fingerprint_score = len(np.intersect1d(peaks_long[:, 1], peaks_short[:, 1])) / len(peaks_short)

    # --- STRATEGY 2: FFT Cross-Correlation (Precise Alignment) ---
    # Fast Fourier Transform correlation is 100x faster for long files
    # We use a reduced sample for speed if files are huge

    # 1. Perform the FFT convolution
    corr = scipy.signal.fftconvolve(y_long, y_short[::-1], mode='valid')
    best_idx = np.argmax(corr)
    offset_seconds = best_idx / sr

    # 2. Extract the specific window from the long file that matched
    y_long_window = y_long[best_idx : best_idx + len(y_short)]

    # 3. Calculate Norms with a safety epsilon
    # Epsilon (1e-9) prevents division by zero if audio is silent
    norm_long = np.linalg.norm(y_long_window)
    norm_short = np.linalg.norm(y_short)

    # print(f"norm_long={norm_long} -- y_long={y_long} -- y_short={y_short}")
    # 4. Robust Correlation Score Calculation
    if norm_long > 0 and norm_short > 0:
        # Normalized Cross-Correlation (NCC) formula
        correlation_score = corr[best_idx] / (norm_long * norm_short)
        # print("if norm_long > 0 and norm_short > 0 - before setting np.clip correlation_score=",correlation_score)
        
        # Optional: Clip the result between -1 and 1 
        # (FFT precision can sometimes push it to 1.0000000001)
        correlation_score = np.clip(correlation_score, -1.0, 1.0)
        # print("if norm_long > 0 and norm_short > 0 - after setting np.clip correlation_score=",correlation_score)

    else:
        # If either is pure silence, the correlation is mathematically undefined (set to 0)
        correlation_score = 0.0

    # 5. Final Sanitization (Double check for any remaining NaN/Inf)
    if not np.isfinite(correlation_score):
        correlation_score = 0.0

    # --- STRATEGY 3: Neural Embedding (Semantic Similarity) ---
    # Using a simple MFCC-based feature vector as a proxy for neural embedding 
    # (Note: For 5070, use 'laion/clap-htsat-fused' for true Neural matching)
    mfcc_long = librosa.feature.mfcc(y=y_long[best_idx:best_idx+len(y_short)], sr=sr, n_mfcc=13).mean(axis=1)
    mfcc_short = librosa.feature.mfcc(y=y_short, sr=sr, n_mfcc=13).mean(axis=1)
    # Cosine similarity is more robust than Pearson for audio features
    neural_proxy_score = 1 - cosine(mfcc_long, mfcc_short)

    # --- FINAL WEIGHTED DECISION ---
    # Landmarks are weighted highest because they resist noise best
    final_score = (fingerprint_score * 0.5) + (correlation_score * 0.2) + (neural_proxy_score * 0.3)
    print(f"after final_score fingerprint_score={fingerprint_score} -- correlation_score={correlation_score} --neural_proxy_score={neural_proxy_score} --final_score={final_score} ")
    neural_proxy_score = float(neural_proxy_score)  if math.isfinite(neural_proxy_score) else 0.0
    results = {
        "match_type": "NO_MATCH",
        "match_score": float(final_score) if math.isfinite(final_score) else 0.0,
        "offset_seconds": float(offset_seconds) if math.isfinite(offset_seconds) else 0.0,
        "confidence": "LOW",
        "details": {
            "fingerprint": float(fingerprint_score)  if math.isfinite(fingerprint_score) else 0.0,
            "alignment": float(correlation_score)  if math.isfinite(correlation_score) else 0.0,
            "semantic": neural_proxy_score
        }
    }
    def sanitize(value):
        if isinstance(value, float) and (np.isinf(value) or np.isnan(value)):
            return 0.0  # Or 1.0 depending on your logic
        return value
    
    # Apply to the dictionary
    results = {k: sanitize(v) for k, v in results.items()}
    # If you have nested dicts (like 'detailed_scores'):
    if results["match_score"] in results:
        results["match_score"] = {k: sanitize(v) for k, v in results["match_score"].items()}

    # Modern Weighted Scoring
    # If Semantic is extremely high, we weigh it more heavily to prevent "Zero-outs"
    # semantic_score = results["match_score"]
    # if semantic_score  > 0.95:
    #     final_score = (fingerprint_score * 0.2) + (correlation_score * 0.1) + (semantic_score * 0.7)
    # else:
    #     final_score = (fingerprint_score * 0.5) + (correlation_score * 0.2) + (semantic_score * 0.3)

    # # Ensure match_score is never exactly 0 if a semantic match was found
    # final_score = max(final_score, semantic_score * 0.5)

    if final_score > 0.85:
        results.update({"match_type": "HIGH_FIDELITY_MATCH", "confidence": "VERY_HIGH"})
    elif final_score > 0.6:
        results.update({"match_type": "ROBUST_PARTIAL_MATCH", "confidence": "MEDIUM"})

    return results