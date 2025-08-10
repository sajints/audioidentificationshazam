import librosa
import librosa.display

def create_spectrogram(times,frequencies, Sxx_db,plt,peaks,sr, hop_length):
    # Extract time and frequency of peaks for plotting
    time_peaks = [times[t] for t, f in peaks]
    freq_peaks = [frequencies[f] for t, f in peaks]

    # Plot with peaks
    plt.figure(figsize=(18, 5))
    librosa.display.specshow(Sxx_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log", cmap="inferno")
    plt.scatter(time_peaks, freq_peaks, color='red', marker="x", label='Detected Peaks')
    plt.legend()
    plt.title("Spectrogram with Detected Peaks")
    plt.show()