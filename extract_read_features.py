import librosa
import os
import glob
import csv
import np
import scipy.io.wavfile
from matplotlib.pyplot import specgram

def write_feat(feat, fn):
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".feat"
    np.save(data_fn, feat)

def create_features(genre_list, base_dir):
    for label, genre in enumerate(genre_list):
        genre_dir = os.path.join(base_dir, genre, "*.wav")
        file_list = glob.glob(genre_dir)
        for fn in file_list:
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
            data.append(genre_list[label])
            write_feat(data,fn)
    return

def read_feat(genre_list, base_dir):
    for label, genre in enumerate(genre_list):
        for fn in glob.glob(os.path.join(base_dir, genre, "*.feat.npy")):
            feat = np.load(fn)
            feat = list(feat)
            feat_to_csv(feat)
    return

def feat_to_csv(data):
    with open('data.csv', mode='a') as csv_writer:
        writer = csv.writer(csv_writer)
        data = list(data)
        writer.writerow(data)
    csv_writer.close()

#Dir1 = '/Users/macbook/Desktop/Machine Learning/Final Project/music(wav)'
Dir2 = '/Users/macbook/Desktop/Machine Learning/Final Project/music_feat'
#create_features(["blues","classical", "country","disco","hiphop","jazz","metal", "pop", "reggae", "rock"],Dir1)
read_feat(["blues","classical", "country","disco","hiphop","jazz","metal", "pop", "reggae", "rock"],Dir2)
