import sounddevice as sd
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from functions import ext_param, npc_predict

from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
import time
import soundfile as sf
import matplotlib.pyplot as plt
from tkinter import *
import numpy as np
import librosa
from IPython.lib.display import Audio
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

N1=30
N2=30
X=np.zeros([N1+N2,28])
y=np.zeros(N1+N2)
#Starting recording for class1
for i in range(N1):
    fs = 8000
    duration = 1
    print('signal class 1:  ',i)
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sd.stop()
    myrecording = myrecording.reshape(len(myrecording))
    f, Sx = signal.welch(x=myrecording, fs=fs, window='hamming', nfft=1000, nperseg=1000)
    v=ext_param(myrecording,Sx,f)
    X[i,:]=v
    y[i]=0

print('second class')
for i in range(N2):
    print('signal class 2:  ', i)
    fs = 8000
    duration = 1
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sd.stop()
    myrecording = myrecording.reshape(len(myrecording))
    f, Sx = signal.welch(x=myrecording, fs=fs, window='hamming', nfft=1000, nperseg=1000)
    v=ext_param(myrecording,Sx,f)
    X[i+N1,:]=v
    y[i+N1]=1
print(X.shape)
print(y.shape)


#here I have X and y
#part 1: Normalise the data
#1- using #x-->(X-mean)/std(X)
data_scaler = StandardScaler()
data_scaler.fit(X)
scaled_data = data_scaler.transform(X)
"""
#2- using #xx-->(x-xmin)/(xmax-xmin)
data_scaler = MinMaxScaler()
data_scaler.fit(X)
scaled_data = data_scaler.transform(X)
"""

#Part 2: Make the PCA Decomposition
pca = PCA(n_components = X.shape[1])
pca.fit(scaled_data)
Xprojected= pca.transform(scaled_data)
"""
print('variances=')
print(pca.explained_variance_) # lambda or eigen values sorted
print ('ratio=')
print(pca.explained_variance_ratio_) #percemtage of each lambda
print ('mean=')
print(pca.mean_) #mean of the data
print ('Covariance=')
print(pca.get_covariance()) #variance covarianec matrix
print ('eigen vectors=')
print(pca.components_) #eigen vectors
print ('sum of remaining variances=')
print(pca.noise_variance_) #sum of remaining eigen value su
"""
#Part 3: Choose the components that fits to 90% of variances
v=pca.explained_variance_ratio_
plt.plot(v)
plt.show()
cv=v.cumsum()
plt.plot(cv)
plt.show()
nb=np.where(cv>=0.90)
n=nb[0][1]
print('n=',n)
X=Xprojected[:,0:n]
#part 4: Divide the data into 30% test and 70% train

X_train, X_test, y_train, y_test = (
    train_test_split(X, y,stratify=y, test_size=0.3))

#part 5- make the classification and print the confusion matrix using
#-5-1: perceptron
clf = Perceptron(tol=1e-15,eta0=0.001,max_iter=10000)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print('confusion matrix using percepron:\n',confusion_matrix(y_test, predictions))
score = clf.score(X_test, y_test)
print('score using perceptron:', score)

#-5-2- FF using dense function
input_shape = [X_train.shape[1]]
Nbclasses=np.max(y)+1
clf = models.Sequential()
clf.add(Dense(6, activation='sigmoid',input_shape=input_shape))
clf.add(Dense(units=8, activation='sigmoid')),
clf.add(Dense(units=3, activation='tanh')),
clf.add(Dense(units=7, activation='linear')),
clf.add(Dense(Nbclasses, activation='softmax'))
clf.summary()
clf.compile(optimizer='rmsprop',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
clf.fit(X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=300,  # total epoch
        verbose=0,
         )
pred = clf.predict(X_test)
cpwi=pred.argmax(axis=1)
print('confusion matrix using FF dense:\n',confusion_matrix(y_test, cpwi))
test_loss= clf.evaluate(X_test, y_test)
print('loss result using FF dense=',test_loss)

#5-3: FF using MLP function
clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(3,3,10,10),
                    activation='tanh', #'logistic','tanh','logistic'},
                    alpha=1e-5,
                    max_iter=40,
                 )
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print('confusion matrix using FF MLP:\n',
      confusion_matrix(y_test, predictions))
score = clf.score(X_test, y_test)
print(score)

#5-4: SVM-RBF
clf = svm.SVC(kernel='rbf',gamma=0.1, C=10,probability=True)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print('confusion matrix using SVM rbf:\n',confusion_matrix(y_test, predictions))
score = clf.score(X_test, y_test)
print(score)

#5-4: SVM- Poly
clf = svm.SVC(kernel='poly',gamma=0.1, C=10,probability=True)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print('confusion matrix using SVM poly:\n',confusion_matrix(y_test, predictions))
score = clf.score(X_test, y_test)
print(score)

#5-4: SVM-linear
clf = svm.SVC(kernel='linear',gamma=0.1, C=10,probability=True)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print('confusion matrix using SVM linear:\n',confusion_matrix(y_test, predictions))
score = clf.score(X_test, y_test)
print(score)

#5-5- KNN
clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print('confusion matrix using KNN:\n',
      confusion_matrix(y_test, predictions))

#5-6 Naive bayes classifier
clf=GaussianNB()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print('confusion matrix using Naive Bayed Classifier:\n',
      confusion_matrix(y_test, predictions))

#5-7: Parzen method
predictions,pwx=npc_predict(X_test,X_train,y_train)
print('confusion matrix using Parzen estimation:\n',
      confusion_matrix(y_test, predictions))


""""
#test new data
print('speak for a new signal:')
fs = 8000
duration = 1
myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
sd.stop()
myrecording = myrecording.reshape(len(myrecording))
f, Sx = signal.welch(x=myrecording, fs=fs, window='hamming', nfft=1000, nperseg=1000)
v=ext_param(myrecording,Sx,f)

scaled_v = data_scaler.transform(v)
scaled_v=scaled_v.reshape(1,-1)

v_projected= pca.transform(scaled_v)
print(scaled_v.shape)
vf=v_projected[:,0:n]

predictions = clf.predict(vf)
print(predictions)

if predictions==1:
    print('backword')
else:
    print('forward')

#np.save('DataMK',X)
#np.save('DataMK',y)
#X=np.load('DataMK.npy')
#y=np.load('DataMK.npy')
"""


