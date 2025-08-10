from scipy import signal
import numpy as np

def detect_peaksnew(Sxx_db, threshold=-40):
    # Collect (time_idx, freq_idx) pairs of peaks
    peak_coords = []

    # Loop through frequency bins
    for freq_idx in range(Sxx_db.shape[0]):
        row = Sxx_db[freq_idx, :]  # One frequency over time (1D)

        # Only keep rows that are strong enough (optional)
        if np.max(row) < threshold:
            continue

        peaks = signal.find_peaks_cwt(row, np.arange(1, 10))
        for time_idx in peaks:
            peak_coords.append((time_idx, freq_idx))  # Time x Frequency

    return np.array(peak_coords)

