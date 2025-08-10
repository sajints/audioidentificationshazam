import librosa
import numpy as np
import hashlib
from scipy import signal

def detect_peaksnew(Sxx_db, threshold=-40):
    peak_coords = []
    for freq_idx in range(Sxx_db.shape[0]):
        row = Sxx_db[freq_idx, :]
        if np.max(row) < threshold:
            continue
        peaks = signal.find_peaks_cwt(row, np.arange(1, 10))
        for time_idx in peaks:
            peak_coords.append((time_idx, freq_idx))
    return np.array(peak_coords)

def generate_fingerprints(peaks, fan_value=5):
    fingerprints = []
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
    y, sr = librosa.load(audio_path, duration=7, sr=44100)
    Sxx = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024))
    Sxx_db = librosa.amplitude_to_db(Sxx, ref=np.max)
    peaks = detect_peaksnew(Sxx_db)
    fingerprints = generate_fingerprints(peaks)
    return fingerprints
