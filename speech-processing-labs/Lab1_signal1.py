import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile,loadmat
from scipy.stats import multivariate_normal

"""
#Part1 Generation of function, signals
t = np.arange(0, 11,0.1)
print(t)
x = (0.85) ** t
plt.figure(figsize = (6,6)) # set the size of figure
# 1. Plotting Analog Signal
plt.subplot(2, 1,1)
plt.title('Analog Signal', fontsize=20)
plt.plot(t, x, linewidth=1, label='x(t) = (0.85)^t')
plt.xlabel('t' , fontsize=15)
plt.ylabel('amplitude', fontsize=15)
plt.legend(loc='upper right')
#plt.show()

# 2. Sampling and Plotting of Sampled signal
plt.subplot(2, 1, 2)
plt.title('Sampling', fontsize=10)
#plt.plot(t, x, linewidth=3, label='x(t) = (0.85)^t')
n = t
markerline, stemlines, baseline = plt.stem(n, x, label='x(n) = (0.85)^n')
plt.setp(stemlines, 'linewidth', 1)
plt.xlabel('n' , fontsize = 45)
plt.ylabel('amplitude', fontsize = 15)
plt.legend(loc='upper right')
plt.show()
"""
"""
#part 2 Generation of signals, pulse sine, exp
impulse = signal.unit_impulse(11, 'mid')
shifted_impulse = signal.unit_impulse(7, 2)
print(impulse)
print(shifted_impulse)

# Sine wave
t = np.linspace(0, 2, 1000)
amp = 5 # Amplitude
f = 10
x = amp * np.sin(2 * np.pi * f * t)

# Exponential Signal
x_ = amp * np.exp(-t)

plt.figure(figsize=(7, 5))

plt.subplot(2, 2, 1)
#plt.plot(np.arange(-5, 5), impulse, linewidth=3, label='Unit impulse function')
markerline, stemlines, baseline = plt.stem(np.arange(-5, 6), impulse)

plt.ylim(-0.1,1.1)

plt.xlabel('time.', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.legend(fontsize=10, loc='lower right')

plt.subplot(2, 2, 2)
plt.plot(shifted_impulse, linewidth=3, label='Shifted Unit impulse function')

plt.xlabel('time.', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.legend(fontsize=10, loc='upper right')

plt.subplot(2, 2, 3)
plt.plot(t, x, linewidth=1, label='Sine wave')
#markerline, stemlines, baseline = plt.stem(t, x)
plt.xlabel('time.', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.legend(fontsize=10, loc='upper right')

plt.subplot(2, 2, 4)
plt.plot(t, x_, linewidth=3, label='Exponential Signal')

plt.xlabel('time.', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.legend(fontsize=10, loc='upper right')

plt.show()
"""
"""
#part 3: Square wave
plt.figure(figsize=(4, 5))
t = np.linspace(0, 1, 500, endpoint=True)
print(t)
plt.plot(t, signal.square(2 * np.pi * 10 * t, duty=0.5))
plt.ylim(-1.5, 1.2)
plt.figure()
sig = np.sin(2 * np.pi * t/2)
pwm = signal.square(2 * np.pi * 30 * t, duty=(sig + 1)/2)
plt.subplot(2, 1, 1)
plt.plot(t, sig)
plt.subplot(2, 1, 2)
plt.plot(t, pwm)
plt.ylim(-1.5, 1.5)
plt.show()
"""
"""

########################
#Part 4 Sampling and Nyquist theorem fs>2fmax

plt.figure(figsize=(6, 4))
plt.suptitle("Sampling a Sine Wave of Fmax=20Hz at 2000Hz then 35 Hz", fontsize=20)

s_rate = 2000 # Hz. Here the sampling frequency is less than the requirement of sampling theorem
T = 1 / s_rate
t = np.arange(0, 0.5,T)
f = 20 # Hz
x1 = np.sin(2 * np.pi * f * t)

plt.subplot(2, 1, 1)
plt.plot(t, x1, label='SineWave of frequency 20 Hz, sampled at 2000Hz')
#markerline, stemlines, baseline = plt.stem(t, x1, label='shanon respected')

plt.xlabel('time.', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.legend(fontsize=10, loc='upper right')

s_rate =35# Hz. Here the sampling frequency is less than the requirement of sampling theorem
T = 1 / s_rate
t1 = np.arange(0, 0.5,T)
x2 = np.sin(2 * np.pi * f * t1) # Since for sampling t = nT.

plt.subplot(2, 1, 2)
plt.plot(t1, x2, 'r-', label='f=20 Hz Sampled at at fs=35Hz')
#markerline, stemlines, baseline = plt.stem(t1, x2, label='shanon not respected')

plt.xlabel('time.', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.legend(fontsize=10, loc='upper right')

plt.show()
"""
"""
###############################
#part 5: Fast Fourier Transform
# Create synthetic signal
fs=1000
dt = 1/fs
t = np.arange(0, 1, dt)
x = np.sin(2 * np.pi * 20* t)+ np.sin(2 * np.pi * 100 * t) # Sum of 2 Sequencies
xnoisy = x + 0.5* np.random.randn(len(x)) # Add some noise
min_signal, max_signal = xnoisy.min(), xnoisy.max()


n = len(x)
freq = np.linspace(-fs/2,fs/2,n)
fhat = np.fft.fft(xnoisy, n)   # Compute the FFT
fhat1=np.fft.fftshift(fhat)
Xf=abs(fhat1)

# frequency array
fig, axs = plt.subplots(2, 1)
plt.sca(axs[0])
plt.plot(t, xnoisy, color='r', linewidth=1.5, label='Noisy')
plt.plot(t, x, color='k', linewidth=2, label='Clean')
plt.xlim(t[0], t[-1])
plt.xlabel('t axis')
plt.ylabel('Vols')
plt.legend()

plt.sca(axs[1])
plt.plot(freq, Xf, color='c', linewidth=2, label='PSD Noisy')
#plt.xlim(freq[idxs_half[0]], freq[idxs_half[-1]])
plt.xlabel('f axis')
plt.ylabel('Power frequency')
plt.legend()
plt.show()
"""
"""
###############################
#Part 6: Spectrogram
fs=1000
dt = 1/fs
t = np.arange(0, 2, dt)
f0=100
t1 = 2
#x=np.cos(2 * np.pi *100*t)
x = np.cos(2 * np.pi * f0* np.power(t, 2))

# plt.rcParams['figure.figsize'] = [8,6]
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(t,x)
plt.subplot(2, 1, 2)
plt.specgram(x, NFFT=256, Fs=fs, noverlap=225, window=np.hamming(256),cmap='jet')
plt.xlabel('Time')
plt.ylabel('Frequency')
#plt.colorbar()
plt.show()

#reading mat file then spectrogram
#sampFreq, sound = wavfile.read('noise_a3s.wav')
S=loadmat('word_matlab.mat')
#S=loadmat('handel1.mat')
Fs=S['Fs']
X=S['mtlb']
#X=S['y']
X=X.reshape(len(X))
print(X.shape)
t=np.arange(0,len(X)/Fs,1/Fs)
plt.figure(figsize=(10, 8))
plt.suptitle("Matlabword", fontsize=20)
plt.subplot(2, 1, 1)
plt.plot(t,X)
plt.subplot(2, 1, 2)
plt.specgram(X, NFFT= 256, noverlap=250, Fs = float(Fs),window=np.hamming(256),cmap='jet')
plt.xlabel('Time')
plt.ylabel('Frequency')
#plt.colorbar()
plt.show()
"""
"""
#Part 7: Periodogram :Sx(f) using signal.welch
fs=1000
dt = 1/fs
t = np.arange(0, 1, dt)
x = np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 80 * t) # Sum of 2 Sequencies
xnoisy = x + 1 * np.random.randn(len(x)) # Add some noise
f, Sxf = signal.welch(xnoisy, fs,window= 'hamming' ,noverlap=450,nfft=500,nperseg=500, scaling='spectrum')
plt.figure()
plt.subplot(221)
plt.plot(t,x)
plt.subplot(222)
plt.plot(t,xnoisy)
plt.xlabel('t')
plt.ylabel('Noisy signal')

specf=np.fft.fftshift(np.fft.fft(xnoisy,len(x)))
plt.subplot(223)
f1=np.linspace(-fs/2,fs/2,len(xnoisy))
plt.plot(f1,np.abs(specf))
plt.xlabel('frequency [Hz]')
plt.ylabel('Fourier Transform')
plt.subplot(224)
plt.plot(f, np.sqrt(Sxf))
plt.xlabel('frequency [Hz]')
plt.ylabel('Power Spectral Density')
plt.show()
"""
"""

#Part 8: Filtering signals using signal.butter

#fs=1000
#dt = 1/fs
#time = np.arange(0, 1, dt)

order = 5
sampling_freq = 1000
cutoff_freq1 = 30
cutoff_freq2 = 90
sampling_duration = 1 #1 seconde
number_of_samples = sampling_freq * sampling_duration
time = np.linspace(0, sampling_duration, number_of_samples, endpoint=False)
x = np.sin(2*np.pi*10*time) + 0.5*np.cos(2*np.pi*60*time) + 1.5*np.sin(2*np.pi*120*time)
normalized_cutoff_freq1 = cutoff_freq1 / (sampling_freq/2)
normalized_cutoff_freq2 = cutoff_freq2 / (sampling_freq/2)
#numerator_coeffs, denominator_coeffs = signal.butter(order, normalized_cutoff_freq2,'high')
#numerator_coeffs, denominator_coeffs = signal.butter(order, normalized_cutoff_freq1,'low')
numerator_coeffs, denominator_coeffs = signal.butter(order, [normalized_cutoff_freq1, normalized_cutoff_freq2],'bandpass')
filtered_signal = signal.lfilter(numerator_coeffs, denominator_coeffs, x)
plt.plot(time, x, 'b-', label='x')
plt.plot(time, filtered_signal, 'g-', linewidth=2, label='filtered signal')
plt.legend()
plt.show()


#Another way
fs=1000
dt = 1/fs
time = np.arange(0, 1, dt)
f1=30/(fs/2)
f2=90/(fs/2)
x = np.sin(10*2*np.pi*time) + 0.5*np.cos(60*2*np.pi*time) + 1.5*np.sin(120*2*np.pi*time)
#[b,a]=signal.butter(5,[f1,f2],'bandpass')
#[b,a]=signal.butter(5,[f2],'high')
[b,a]=signal.butter(5,[f1],'low')
y=signal.lfilter(b,a,x)
plt.subplot(211)
plt.plot(time,x)
plt.subplot(212)
plt.plot(time,y)
plt.show()

"""

#Part 9: Convolutuon- Correlation
fs=3000
dt = 1/fs
t = np.arange(0, 2, dt)
x = np.sin(2 * np.pi * 150 * t) #+ np.sin(2 * np.pi * 120 * t) # Sum of 2 Sequencies
xnoisy = x + 1* np.random.randn(len(x)) # Add some noise

plt.figure(figsize=(8, 8))
plt.suptitle("SineWave Plus noise and its correlation", fontsize=20)

plt.subplot(2, 2, 1)
plt.plot(t, xnoisy, 'm', label='Signal+noise')
plt.xlabel('time.', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.legend(fontsize=10, loc='upper right')

plt.subplot(2, 2, 2)
n = len(x)
Xf=abs(np.fft.fftshift(np.fft.fft(xnoisy,len(t))))
freq = np.linspace(-fs/2,fs/2,n)   # frequency array
plt.plot(freq, Xf, color='c', linewidth=2, label='FFT of the Noisy Signal')
#plt.xlim(freq[idxs_half[0]], freq[idxs_half[-1]])
plt.xlabel('f axis')
plt.ylabel(' FFT graph')
plt.legend()

plt.subplot(2, 2, 3)
corr = signal.correlate(xnoisy,xnoisy,'full')
lags = signal.correlation_lags(len(xnoisy),len(xnoisy)) #tau
plt.plot(lags,corr, 'm', label='Correlation')

plt.subplot(2, 2, 4)
n = len(corr)
PSD=abs(np.fft.fftshift(np.fft.fft(corr,n)))
freq = np.linspace(-fs/2,fs/2,n)   # frequency array
plt.plot(freq, PSD, color='c', linewidth=2, label='PSD of the Autocorr')
#plt.xlim(freq[idxs_half[0]], freq[idxs_half[-1]])
plt.xlabel('f axis')
plt.ylabel('Power SD')
plt.legend()

plt.show()




#Part 10- Histogram

# Creating dataset
np.random.seed(23685752)
N_points = 10000
n_bins = 20
# Creating distribution
x = np.random.randn(N_points)

# Creating histogram
fig, axs = plt.subplots(2, 1,
                        figsize=(10, 7),
                        tight_layout=True)
plt.sca(axs[1])
plt.hist(x, bins=n_bins)
# Show plot
plt.sca(axs[0])
plt.plot(x)
plt.show()


"""
#part 11: Multivariate distribution

# Initializing the random seed
random_seed = 1000

# List containing the variance
# covariance values
cov_val = [-0.8, 0, 0.8]

# Setting mean of the distributino to
# be at (0,0)
mean = np.array([0, 0])

# Iterating over different covariance
# values
for idx, val in enumerate(cov_val):
    plt.subplot(1, 3, idx + 1)

    # Initializing the covariance matrix
    cov = np.array([[1, val], [val, 1]])

    # Generating a Gaussian bivariate distribution
    # with given mean and covariance matrix
    distr = multivariate_normal(cov=cov, mean=mean,
                                seed=random_seed)
    # Generating 5000 samples out of the
    # distribution
    data = distr.rvs(size=5000)

    # Plotting the generated samples
    plt.plot(data[:, 0], data[:, 1], 'o', c='lime',
             markeredgewidth=0.5,
             markeredgecolor='black')
    plt.title(f'Covariance between x1 and x2 = {val}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axis('equal')

plt.show()

fig = plt.figure()

# Initializing the random seed
random_seed = 1000

# List containing the variance
# covariance values
cov_val = [-0.8, 0, 0.8]

# Setting mean of the distributino
# to be at (0,0)
mean = np.array([0, 0])

# Storing density function values for
# further analysis
pdf_list = []

# Iterating over different covariance values
for idx, val in enumerate(cov_val):

    # Initializing the covariance matrix
    cov = np.array([[1, val], [val, 1]])

    # Generating a Gaussian bivariate distribution
    # with given mean and covariance matrix
    distr = multivariate_normal(cov=cov, mean=mean,
                                seed=random_seed)

    # Generating a meshgrid complacent with
    # the 3-sigma boundary
    mean_1, mean_2 = mean[0], mean[1]
    sigma_1, sigma_2 = cov[0, 0], cov[1, 1]

    x = np.linspace(-3 * sigma_1, 3 * sigma_1, num=100)
    y = np.linspace(-3 * sigma_2, 3 * sigma_2, num=100)
    X, Y = np.meshgrid(x, y)

    # Generating the density function
    # for each point in the meshgrid
    pdf = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pdf[i, j] = distr.pdf([X[i, j], Y[i, j]])

    # Plotting the density function values
    key = 131 + idx
    ax = fig.add_subplot(key, projection='3d')
    ax.plot_surface(X, Y, pdf, cmap='viridis')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f'Covariance between x1 and x2 = {val}')
    pdf_list.append(pdf)
    ax.axes.zaxis.set_ticks([])

plt.tight_layout()
plt.show()

# Plotting contour plots
for idx, val in enumerate(pdf_list):
    plt.subplot(1, 3, idx + 1)
    plt.contourf(X, Y, val, cmap='viridis')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f'Covariance between x1 and x2 = {cov_val[idx]}')
plt.tight_layout()
plt.show()
"""