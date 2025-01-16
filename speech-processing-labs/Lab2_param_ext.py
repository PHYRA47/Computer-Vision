
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal,stats
import math
import sounddevice as sd

def ext_param(x,Sx,f): #28 parameters

    Df=f[2]-f[1] #f[n+1]-f[n]
    param=[]

    #Statistical paramameters related to probability f(x)
    param.append(stats.skew(x)) #M3
    param.append(stats.kurtosis(x))#M4

    #Spectral Parameters related to PSD Sx(f)
    #1power of the signal
    M0=Sx.sum() # integration of Sx(f) over all frequencies for M0
    param.append(M0)

    #2MPF, #Mean frequency
    M1=sum(f*Sx) # integration of f*Sx(f) over all frequencies for M1
    mpf=M1/M0 # 
    param.append(mpf)

    # Skewness: calcul de coefficient de dissymetry
    cd1= sum((f-mpf)**3*Sx)
    cd2=sum((f-mpf)**2*Sx)
    Cd=cd1/math.sqrt(cd2**3)
    param.append(Cd)

    # kurtosis frequency
    ca1= sum((f-mpf)**4*Sx) # M* for sum((f-mpf)*Sx) and M for sum(f*Sx)
    ca2= sum((f-mpf)**2*Sx)
    Ca=ca1/(ca2**2)
    param.append(Ca)

    #median of frequency Fmed
    sc = Sx.cumsum()
    hs = Sx.sum() / 2
    fm = np.where(sc >= hs)
    fmed =fm[0][1] * Df
    param.append(fmed)

    # peak of frequency
    pf =np.array(Sx).argmax()
    pf=(pf)*Df # pf may have multiple values (equal maximums) so we take the first one
    param.append(pf)

    
    # relative energy by frequency band 10
    w=[] 
    Len1=len(Sx)
    Lseg=round(Len1/10)
    for i in range(10):#0...9
        w=sum(Sx[Lseg*i:Lseg*(i+1)-1])
        w = w / M0
        param.append(w)

    # deciles (k=0.1) or quartiles (k=0.25) 9
    k = 0.1
    sc = Sx.cumsum()
    surfc = k * Sx.sum()
    for i in range(int(1/k)-1): #0...8
        fm = np.where((sc>=(i+1)*surfc))
        fd = fm[0][1] * Df
        param.append(fd)

    # entropie
    ent=-sum(Sx*np.log(Sx))
    param.append(ent)

    param=np.array(param)
    #param1=param[[3,8,9,10,18,19,20,27]]

    return param # return the 28 parameters

"""
#first example Sine waves
# plus noise signal simulation
fs=2000
dt = 1/fs
t = np.arange(0, 1, dt)
x = 2*np.sin(2 * np.pi * 35 * t) + 1*np.sin(2 * np.pi * 380 * t)+1*np.sin(2 * np.pi * 135 * t)# Sum of 3 Sequencies
xnoisy = x + 1* np.random.randn(len(x)) # Add some noise

f, Sx = signal.welch(xnoisy, fs, window='hamming', noverlap=250, nfft=256, nperseg=256, scaling='spectrum')

param=ext_param(xnoisy,Sx,f)

print(param)
print(len(param))

plt.figure()
plt.subplot(211)
plt.plot(t,xnoisy)
plt.subplot(212)
plt.plot(f,Sx)
plt.show()
"""

"""
#second part: creating a signal from white noise
#having a band of frequency

# frequency sampling
fs=2000

# bandpass filter parameters
fc1=300
fc2=600

[b,a]=signal.butter(2,[fc1/(fs/2),fc2/(fs/2)],'bandpass') # bandpass filter
v=np.random.randn(5000) # white noise

y=signal.lfilter(b,a,v) # bandpass signal
y = y / y.std() # normalization is optional

f, Sy = signal.welch(y, fs, window='hamming', noverlap=255, nfft=256, nperseg=256, scaling='spectrum')

param=ext_param(y,Sy,f)

print('parametres:')
print(param)
plt.figure()
plt.subplot(211)
plt.plot(y)
plt.subplot(212)
plt.plot(f,Sy)
print(len(param))
plt.show()

"""
# """
#third example: acquiring real voice then calculate the parameters
fs = 8000
duration = 2 #in sec
myrecording = sd.rec(frames=duration*fs, samplerate=fs, channels=1) # channels=1 for internal microphone
sd.wait()
sd.stop()
x= myrecording.reshape(len(myrecording))
plt.plot(x)
plt.show()

f, Sx = signal.welch(x, fs, window='hamming', noverlap=250, nfft=256, nperseg=256, scaling='spectrum')

param=ext_param(x,Sx,f)
print('parametres:')
print(param)
plt.figure()
plt.subplot(211)
plt.plot(x)
plt.subplot(212)
plt.plot(f,Sx)
print(len(param))
plt.show()
# """