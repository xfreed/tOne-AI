"""
-*- coding: utf-8 -*-
"""
import json
import logging
import os
#import time
import warnings


import librosa
import numpy as np
import pandas as pd
import sklearn.preprocessing
import keras
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D as Conv
from keras.layers.convolutional import MaxPooling2D as Pool
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.regularizers import l2 as L2

import pydub
from tqdm import tqdm
from config import *

THEANO_FLAGS = ('device=gpu0,'
                'floatX=float32,'
                'dnn.conv.algo_bwd_filter=deterministic,'
                'dnn.conv.algo_bwd_data=deterministic')

os.environ['THEANO_FLAGS'] = THEANO_FLAGS
os.environ['KERAS_BACKEND'] = 'theano'

keras.backend.set_image_dim_ordering('th')




def to_one_hot(targets, class_count):
    """Encode target classes in a one-hot matrix.
    """
    one_hot_enc = np.zeros((len(targets), class_count))
    #   HERE IS NEW CODE
    #for r in range(len(targets)):
    #    one_hot_enc[r, targets[r]] = 1
    for value in targets:
        one_hot_enc[value, targets[value]] = 1
    return one_hot_enc


def extract_segment(filename):
    """Get one random segment from a recording.
    """
    spec = np.load('dataset/tmp/' + filename + '.spec.npy').astype('float32')

    offset = np.random.randint(0, np.shape(spec)[1] - SEGMENT_LENGTH + 1)
    spec = spec[:, offset:offset + SEGMENT_LENGTH]

    return np.stack([spec])


def iterrows(dataframe):
    """Iterate over a random permutation of dataframe rows.
    """
    while True:
        for row in dataframe.iloc[np.random.permutation(len(dataframe))].itertuples():
            yield row


def iterbatches(batch_size, training_dataframe):
    """Generate training batches.
    """
    itrain = iterrows(training_dataframe)

    while True:
        x, y = [], []

        for i in range(batch_size):
            row = next(itrain)
            x.append(extract_segment(row.filename))
            y.append(LE.transform([row.category])[0])

        x = np.stack(x)
        y = to_one_hot(np.array(y), len(LABELS))

        x -= AUDIO_MEAN
        x /= AUDIO_STD

        yield x, y


if __name__ == '__main__':
    np.random.seed(1)

    logging.basicConfig(level=logging.DEBUG)
    LOGGER = logging.getLogger(__name__)

    # Load dataset
    META = pd.read_csv('dataset/dataset.csv')
    LABELS = pd.unique(META.sort_values('category')['category'])
    LE = sklearn.preprocessing.LabelEncoder()
    LE.fit(LABELS)

    # Generate spectrograms
    LOGGER.info('Generating spectrograms...')

    if not os.path.exists('dataset/tmp/'):
        os.mkdir('dataset/tmp/')

    for row in tqdm(META.itertuples(), total=len(META)):
        spec_file = 'dataset/tmp/' + row.filename + '.spec.npy'
        audio_file = 'dataset/audio/' + row.filename

        if os.path.exists(spec_file):
            continue

        audio = pydub.AudioSegment.from_file(audio_file).set_frame_rate(SAMPLING_RATE).set_channels(1)# pylint: disable=line-too-long
        audio = (np.fromstring(audio._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)

        spec = librosa.feature.melspectrogram(audio, SAMPLING_RATE, n_fft=FFT_SIZE,
                                              hop_length=CHUNK_SIZE, n_mels=MEL_BANDS)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # Ignore log10 zero division
            spec = librosa.core.perceptual_weighting(spec, MEL_FREQS, amin=1e-5, ref_power=1e-5,
                                                     top_db=None)

        spec = np.clip(spec, 0, 100)
        np.save(spec_file, spec.astype('float16'), allow_pickle=False)

    # Define model
    LOGGER.info('Constructing model...')

    INPUT_SHAPE = 1, MEL_BANDS, SEGMENT_LENGTH

    MODEL = keras.models.Sequential()

    MODEL.add(Conv(80, (3, 3), kernel_regularizer=L2(0.001), kernel_initializer='he_uniform',
                   INPUT_SHAPE=INPUT_SHAPE))
    MODEL.add(LeakyReLU())
    MODEL.add(Pool((3, 3), (3, 3)))

    MODEL.add(Conv(160, (3, 3), kernel_regularizer=L2(0.001), kernel_initializer='he_uniform'))
    MODEL.add(LeakyReLU())
    MODEL.add(Pool((3, 3), (3, 3)))

    MODEL.add(Conv(240, (3, 3), kernel_regularizer=L2(0.001), kernel_initializer='he_uniform'))
    MODEL.add(LeakyReLU())
    MODEL.add(Pool((3, 3), (3, 3)))

    MODEL.add(Flatten())
    MODEL.add(Dropout(0.5))

    MODEL.add(Dense(len(LABELS), kernel_regularizer=L2(0.001), kernel_initializer='he_uniform'))
    MODEL.add(Activation('softmax'))

    OPTIMIZER = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
    MODEL.compile(loss='categorical_crossentropy', OPTIMIZER=OPTIMIZER, metrics=['accuracy'])

    # Train model
    BATCH_SIZE = 100
    EPOCH_MULTIPLIER = 10
    EPOCHS = 1000 // EPOCH_MULTIPLIER
    EPOCH_SIZE = len(META) * EPOCH_MULTIPLIER
    BPE = EPOCH_SIZE // BATCH_SIZE

    #LOGGER.info('Training... (batch size of {} | {} batches per epoch)'.format(BATCH_SIZE, BPE))
    LOGGER.info('Training... (batch size of %s| %sbatches per epoch)', BATCH_SIZE, BPE)

    MODEL.fit_generator(generator=iterbatches(BATCH_SIZE, META),
                        steps_per_epoch=BPE,
                        EPOCHS=EPOCHS)

    with open('model.json', 'w') as file:
        file.write(MODEL.to_json())

    MODEL.save_weights('model.h5')

    with open('model_labels.json', 'w') as file:
        json.dump(LE.classes_.tolist(), file)
