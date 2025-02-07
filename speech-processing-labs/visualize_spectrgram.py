import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import time

Fs = 8000  # Sampling frequency
record_seconds = 2  # Recording duration

input("Press Enter to start recording...")
print("Recording...")
mySpeech = sd.rec(int(record_seconds * Fs), samplerate=Fs, channels=1, dtype='int16')
sd.wait()  # Wait until recording is finished
print("Recording stopped.")

# Play the recorded audio
sd.play(mySpeech, Fs)
time.sleep(record_seconds)
sd.stop()

# Plot the waveform
plt.figure(1)
plt.plot(mySpeech)
plt.title("Waveform of Recorded Speech")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")

# Plot the spectrogram
plt.figure(2)
f, t, Sxx = scipy.signal.spectrogram(mySpeech[:, 0], Fs, window=('kaiser', 5), nperseg=500, noverlap=475)
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.title("Spectrogram of Recorded Speech")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.colorbar(label="Power/Frequency (dB/Hz)")

plt.show()
