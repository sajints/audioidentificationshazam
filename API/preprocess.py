from scipy.signal import butter, lfilter
import noisereduce as nr
import numpy as np
import librosa

def highpass_filter(y, sr, cutoff=80):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, y)

def reduce_noise(y, sr):
    # This automatically finds a silent part of the audio to use as a noise mask
    reduced_noise = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
    return reduced_noise

#preprocess_audio(audio_path)
def preprocess_audio(file_path):
    # 1. Load Audio
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    
    # 2. Apply High-Pass Filter (Remove low-end noise)
    y = highpass_filter(y, sr)
    
    # 3. Reduce Noise (Remove hiss/background hum)
    # Note: Install with 'pip install noisereduce'
    y = reduce_noise(y, sr)
    
    # 4. Normalize (Ensure consistent volume)
    y = normalize_audio(y)
    
    return y, sr

def normalize_audio(y):
    """Normalizes audio to a peak of 1.0"""
    max_val = np.max(np.abs(y))
    if max_val > 0:
        return y / max_val
    return y