import os,sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import librosa
import librosa.display
from Client.peaks import detect_peaksnew # detect_peaks, generate_fingerprints, store_fingerprints, match_fingerprints, detect_peaksnew
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fingerprints import generate_fingerprints, store_fingerprints, match_fingerprints
from sqllite import create_database
from Client.spectrogram import create_spectrogram

#Load an audio file
file_path="./aud.mp3"
#sr-16000 #44100
y, sr = librosa.load(file_path, duration=7, sr=44100)
#Define FFT parameters
n_fft = 4096 #FFT window size
hop_length = 1024 # Overlap between consecutive FFT windows
#Compute the Short-Time Fourier Fransform (STFT)
Sxx = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
#Convert to a decibel scale
Sxx_db = librosa.amplitude_to_db(Sxx, ref=np.max)
#Generate time and frequency axes
times = librosa.times_like(Sxx, sr=sr, hop_length=hop_length)
frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
#Plot spectrogram
plt.figure(figsize=(10, 5))
librosa.display.specshow(Sxx_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log", cmap="inferno")
#Detect peaks
peaks = detect_peaksnew(Sxx_db)

# For debugging purposes, you can plot the spectrogram with detected peaks
# create_spectrogram(times, frequencies, Sxx_db, plt, peaks, sr, hop_length)

# STEP 2: Convert to fingerprints
fingerprints = generate_fingerprints(peaks)

# STEP 3: Store (during registration phase)
store_fingerprints(fingerprints, song_id="aud.mp3")

# STEP 4: Match (during recognition phase)
match = match_fingerprints(fingerprints)
print("Best Match:", match)