import librosa
import os
import glob
import csv
import np
import scipy.io.wavfile
from matplotlib.pyplot import specgram

#fn = music address
fn = '/Users/macbook/Desktop/Machine Learning/Final Project/monologue.wav'
sample_rate, X = scipy.io.wavfile.read(fn)
fft_features = np.mean(abs(scipy.fft(X)[:1000]))
y, sr = librosa.load(fn,mono=True,duration=30)
chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
zcr = np.mean(librosa.feature.zero_crossing_rate(y))
mfcc = librosa.feature.mfcc(y=y, sr=sr)
data = [chroma_stft,spec_cent,spec_bw,rolloff,zcr,fft_features]
for i in mfcc:
    data.append(np.mean(i))
file = open('data_to_predict.csv', 'w', newline='')

with file:
    writer = csv.writer(file)
    writer.writerow(data)

file.close()

