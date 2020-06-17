import librosa
import librosa.display
import pandas as pd
import glob
import csv
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, Model, model_from_json
from process_audio import *
emotions = {}
emotions[0] = 'sad'
emotions[1] = 'happy'
emotions[2] = 'neutral'
emotions[3] = 'angry'

angry_sample = 'angrysample.wav'
neutral_sample = 'neutralsample.wav'
sample_spectrogram = get_fitted_spectrogram(angry_sample)
plt.show()
sample_spectrogram2 = get_fitted_spectrogram(neutral_sample)
plt.show()
input_length = 22500 * 6

json_file = open('weights/v3_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("weights/best_model.hdf5")
l = np.asarray([sample_spectrogram, sample_spectrogram2])
l = l.reshape(np.append(l.shape, 1))
q = loaded_model.predict(l)
guess1 = emotions[np.argmax(q[0])]
guess2 = emotions[np.argmax(q[1])]
print(f'angry sample guess: {guess1}, neutral sample guess: {guess2}')
print(q[0])
print(q[1])
