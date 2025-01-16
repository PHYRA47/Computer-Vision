import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from tkinter import *
import numpy as np
import librosa
from IPython.lib.display import Audio


def Voice_rec():
    fs = 8000
    # seconds
    duration = 3
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sd.stop()
    # Save as FLAC file at correct sampling rate
    sd.play(myrecording,samplerate=fs)
    sd.stop()
    return sf.write('my_Audio_file.flac', myrecording, fs)

def Voice_specgram():
    x,fs=sf.read('my_Audio_file.flac')
    plt.figure()
    plt.subplot(211)
    plt.plot(x)
    newarr = x.reshape(len(x))
    plt.subplot(212)
    plt.specgram(newarr, NFFT=512, noverlap=500, Fs=float(fs), window=np.hamming(512),cmap='jet')
    plt.show()

def Calc_MFCC():
    x, fs = sf.read('my_Audio_file.flac')
    n_fft = int(fs * 0.025)
    # 3. Run the default beat tracker
    mfcc = librosa.feature.mfcc(y=x, sr=fs, hop_length=300, n_mfcc=13,n_fft=n_fft)
    print(mfcc.shape)


    # And the first-order differences (delta features)
    mfcc_delta = librosa.feature.delta(mfcc)
    print(mfcc_delta.shape)

    # Display the MFCCs
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()


master = Tk()

Label(master, text=" Voice Recoder : ").grid(row=0, sticky=W, rowspan=15)

b = Button(master, text="Start", command=Voice_rec)
b.grid(row=30, column=0, columnspan=2, rowspan=2, padx=5, pady=5)

c = Button(master, text="Spectrogram", command=Voice_specgram)
c.grid(row=90, column=0, columnspan=2, rowspan=2, padx=5, pady=5)

d = Button(master, text="MFCC param", command=Calc_MFCC)
d.grid(row=120, column=0, columnspan=2, rowspan=2, padx=5, pady=5)

mainloop()
