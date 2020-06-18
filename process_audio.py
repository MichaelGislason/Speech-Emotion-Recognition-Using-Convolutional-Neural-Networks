import librosa
import librosa.display
import pandas as pd
import glob
import csv
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, Model
from keras.utils import plot_model, to_categorical
from keras.layers import Dense, Input, Activation, Conv2D, MaxPooling2D, concatenate, Flatten, Dropout, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adadelta, Adam
from keras.optimizers import SGD
import random as rand
from matplotlib import pyplot
import math
from sklearn.metrics import classification_report, confusion_matrix
SAMPLE_RATE= 22500

''' our audio files are cut/padded to a fixed input size,
    so our input is '''
input_length = 22500 * 6
N_MELS = 128
N_WINDOWS = 264
rows, cols = N_WINDOWS, N_MELS
num_outcomes = 4
EPOCHS = 20
BATCH_SIZE = 64
x_final_test = []
final_test_paths = []
final_test_label = []

emotions = {}
emotions[0] = 'sad'
emotions[1] = 'happy'
emotions[2] = 'neutral'
emotions[3] = 'angry'



def write_databases_to_csv(emotions=["sad", "neutral", "happy", 'angry'], train_name="TRAIN.csv",
                           test_name="TEST.csv", verbose=1):
    train_target = {"path": [], "emotion": []}
    test_target = {"path": [], "emotion": []}
    i = 0
    for category in emotions:
        # for training speech directory
        for i, path in enumerate(glob.glob(f"data/training/Actor_*/*_{category}.wav")):
            train_target["path"].append(path)
            train_target["emotion"].append(category)
        if verbose:
            print(f"[TESS&RAVDESS] There are {i} training audio files for category:{category}")

        # for validation speech directory
        for i, path in enumerate(glob.glob(f"data/validation/Actor_*/*_{category}.wav")):
            test_target["path"].append(path)
            test_target["emotion"].append(category)
        if verbose:
            print(f"[TESS&RAVDESS] There are {i} testing audio files for category:{category}")
    pd.DataFrame(test_target).to_csv(test_name)
    pd.DataFrame(train_target).to_csv(train_name)


def preprocess_audio_as_mel_spectrogram(audio, sample_rate=22050, window_size=20,  # log_specgram
                           step_size=10, eps=1e-10):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=22050, n_fft=2048, hop_length=512, n_mels=128)
    mel_db = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
    librosa.display.specshow(mel_spec)
    librosa.display.cmap(mel_spec)
    return mel_db.T


def get_fitted_spectrogram(file_path, input_length=input_length):
    data, sr = librosa.load(file_path, sr=None)
    max_offset = abs(len(data) - input_length)
    offset = np.random.randint(max_offset)
    if len(data) > input_length:
        data = data[offset:(input_length + offset)]
    else:
        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

    data = preprocess_audio_as_mel_spectrogram(data)
    return data


def create_parallel_convnet():
    input_shape = Input(shape=(rows, cols, 1))
    tower_1 = Conv2D(16, (12, 16), padding='same', activation='relu')(input_shape)
    tower_1 = MaxPooling2D((6, 8), strides=(1, 1), padding='same')(tower_1)
    tower_2 = Conv2D(16, (18, 24), padding='same', activation='relu')(input_shape)
    tower_2 = MaxPooling2D((9, 12), strides=(1, 1), padding='same')(tower_2)
    tower_3 = Conv2D(16, (24, 32), padding='same', activation='relu')(input_shape)
    tower_3 = MaxPooling2D((12, 16), strides=(1, 1), padding='same')(tower_3)
    tower_4 = Conv2D(16, (30, 40), padding='same', activation='relu')(input_shape)
    tower_4 = MaxPooling2D((15, 20), strides=(1, 1), padding='same')(tower_4)

    merged = concatenate([tower_1, tower_2, tower_3, tower_4], axis=1)
    merged = Flatten()(merged)
    out = Dense(32, activation='relu')(merged)
    out2 = Dense(16, activation='relu')(out)
    out3 = Dense( num_outcomes, activation='softmax')(out2)
    model = Model(input_shape, out3)
    plot_model(model, to_file='modelB.png')
    return model
def create_augmented_parallel_convnet():
    input_shape = Input(shape=(rows, cols, 1))
    tower_1 = Conv2D(16, (12, 16), padding='same', activation='relu')(input_shape)
    tower_1 = MaxPooling2D((6, 8), strides=(1, 1), padding='same')(tower_1)
    tower_2 = Conv2D(16, (18, 24), padding='same', activation='relu')(input_shape)
    tower_2 = MaxPooling2D((9, 12), strides=(1, 1), padding='same')(tower_2)
    tower_3 = Conv2D(16, (24, 32), padding='same', activation='relu')(input_shape)
    tower_3 = MaxPooling2D((12, 16), strides=(1, 1), padding='same')(tower_3)
    tower_4 = Conv2D(16, (30, 40), padding='same', activation='relu')(input_shape)
    tower_4 = MaxPooling2D((15, 20), strides=(1, 1), padding='same')(tower_4)

    merged = concatenate([tower_1, tower_2, tower_3, tower_4], axis=1)
    merged = Flatten()(merged)
    out = Dense(32, activation='relu')(merged)
    normalized = BatchNormalization() (out)
    out2 = Dense(16, activation='relu')(normalized)
    out3 = Dense(num_outcomes, activation='softmax')(out2)
    model = Model(input_shape, out3)
    plot_model(model, to_file='modelC.png')
    return model
def create_sequential_convnet():
    input_shape = Input(shape=(rows, cols, 1))
    tower_1 = Conv2D(16, (12, 16), padding='same', activation='relu')(input_shape)
    tower_1 = MaxPooling2D((6, 8), strides=(1, 1), padding='same')(tower_1)
    flattened = Flatten() (tower_1)
    out = Dense(32, activation='relu')(flattened)
    out2 = Dense(16, activation='relu')(out)
    out3 = Dense(num_outcomes, activation='softmax')(out2)
    model = Model(input_shape, out3)
    plot_model(model, to_file='modelA.png')
    return model

def run_test():
    final_test_paths = []
    final_test_label = []
    
    #databases used were the freely available TESS and RAVDESS
    write_databases_to_csv()
    with open('TRAIN.csv', newline='') as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            l = tuple(row)
            data.append(l[1:])
    f.close()
    data_train = data[1:]
    print(data_train)
    paths = []
    labels = []
    sad, happy, neutral, angry = 0, 0, 0, 0
    for item in data_train:
        paths.append(item[0])
        label = item[1]
        if label == 'sad':
            sad+=1
            labels.append(0)
        elif label == 'happy':
            happy+=1
            labels.append(1)
        elif label == 'neutral':
            neutral+=1
            labels.append(2)
        elif label == "angry":
            angry+=1
            labels.append(3)
    print(f'{sad/len(data_train)}: sad')
    print(f'{happy / len(data_train)}: happy')
    print(f'{neutral / len(data_train)}: neutral')
    print(f'{angry / len(data_train)}: angry')
    with open('TEST.csv', newline='') as foo:
        reader = csv.reader(foo)
        data = []
        for row in reader:
            l = tuple(row)
            data.append(l[1:])
    foo.close()
    data_test = data[1:]
    paths_test = []
    labels_test = []
    for item in data_test:
        path = item[0]
        label = item[1]
        random_num = rand.randint(0, 2)
        if random_num != 2:
            paths_test.append(path)
            if label == 'sad':
                labels_test.append(0)
            elif label == 'happy':
                labels_test.append(1)
            elif label == 'neutral':
                labels_test.append(2)
            elif label == "angry":
                labels_test.append(3)
        else:
            final_test_label.append(label)
            final_test_paths.append(path)
    print(f'There are {len(final_test_paths)}, {len(final_test_label)}')

    x_train = []
    x_test = []
    for recording in paths:
        spec = get_fitted_spectrogram(recording)
        x_train.append(spec)
    for rec in paths_test:
        spec = get_fitted_spectrogram(rec)
        x_test.append(spec)
    for rec in final_test_paths:
        spec = get_fitted_spectrogram(rec)
        x_final_test.append(spec)
    #reshape our inputs as needed (see Keras documentation)
    x_train = np.asarray(x_train)
    x_train = x_train.reshape(np.append(x_train.shape, 1))

    x_test = np.asarray(x_test)
    x_test = x_test.reshape(np.append(x_test.shape, 1))

    x_final_test = np.asarray(x_final_test)
    x_final_test = x_final_test.reshape(np.append(x_final_test.shape, 1))
    #use one-hot encoding
    y_list = to_categorical(labels, 4)
    y_test_binary = to_categorical(labels_test, 4)

    #this was the best performing architecture but other architectures can be substituted in the following line
    model = create_augmented_parallel_convnet()
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['mae', "accuracy"])

    # Save the model
    model_json = model.to_json()
    json_file = open("./weights/v3_model.json", "w+")
    json_file.write(model_json)
    json_file.close()
    trial_num = input('Enter Trial Number: ')
    filepath= f"weights/V3_WEIGHTS_TRIAL_{trial_num}-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]

    # Fit the model weights
    print('fitting model')
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1)
    history =model.fit(x_train, y_list,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            callbacks=callbacks_list,
            validation_data=(x_test, y_test_binary))

    q = model.predict(x_final_test)

    predictions = open(f'{trial_num}_Predictions.txt', 'w+')
    print(q, file=predictions)
    predictions.close()
    foo = open(f'{trial_num}_Final_test_label.txt', 'w+')
    print(final_test_label, file=foo)
    foo.close()
    f00 = open('{trial_num}_Final_Test_Paths.txt', 'w+')
    print(final_test_paths, file=f00)
    f00.close()

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

if __name__ == '__main__':
    run_test()
