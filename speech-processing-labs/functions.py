import math
import numpy as np 
from scipy import signal,stats

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

def npc_predict(X_test,X_train,y_train):
    n_samples=int(X_train.shape[0])
    n_samp_test=int(X_test.shape[0])
    nbclasses = int(y_train.max()+1)

    pwxi = np.zeros([int(n_samp_test), int(nbclasses)])

    pw = np.zeros(nbclasses)
    vol = np.zeros(nbclasses)
    n = np.zeros(nbclasses)
    h = np.zeros(nbclasses)
    K = np.zeros(nbclasses)
    d=X_train.shape[1]

    for i in range(nbclasses):
        n[i] = sum(y_train == i)
        pw[i] = n[i] / n_samples
        vol[i] = 1 / np.sqrt(n[i])
        h[i] = (n[i] ** (-1 / (2 * d)))

    predictions = np.zeros(X_test.shape[0])
    pxw = np.zeros(nbclasses)
    pwx = np.zeros(nbclasses)
    px = 0

    for sample in range(X_test.shape[0]):
        Xs = X_test[sample, :]
        for c in range(nbclasses):
            Xtesti = X_train[y_train == c, :]
            Xtesti1 = (Xtesti - Xs) / h[c]
            val = np.linalg.norm(Xtesti1, axis=1)
            val1 = np.exp(-val ** 2 / 2)
            K[c] = sum(val1)
            pxw[c] = K[c] / (n[c] * vol[c])
            px = px + pxw[c] * pw[c]

        for i in range(nbclasses):
            pwx[i] = pxw[i] * pw[i] / px

        pwxi[sample,:]=pwx/sum(pwx)
        predictions[sample] = int(pwx.argmax())
    return predictions,pwxi