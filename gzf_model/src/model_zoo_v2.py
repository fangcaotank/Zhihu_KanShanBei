import os

import keras
import keras.backend as K
import numpy as np
from keras.layers import recurrent, Dense, Input, Dropout, TimeDistributed, Flatten, concatenate
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Conv1D
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.regularizers import l2

from AttLayer import Attention


def get_embed(tokenizer, GLOVE_STORE, USE_GLOVE, VOCAB, EMBED_HIDDEN_SIZE,
               TRAIN_EMBED, MAX_LEN):
    # GLOVE_STORE = 'precomputed_glove.weights'
    if USE_GLOVE:
        if not os.path.exists(GLOVE_STORE + '.npy'):
            print('Computing GloVe')

            embeddings_index = {}
            with open(r'D:\users\shaohu\data\glove.840B.300d.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split(' ')
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs

            # prepare embedding matrix
            embedding_matrix = np.zeros((VOCAB, EMBED_HIDDEN_SIZE))
            for word, i in tokenizer.word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
                else:
                    print('Missing from GloVe: {}'.format(word))

            np.save(GLOVE_STORE, embedding_matrix)

        print('Loading GloVe')
        embedding_matrix = np.load(GLOVE_STORE + '.npy')

        embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], input_length=MAX_LEN,
                          trainable=TRAIN_EMBED)
    else:
        embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, input_length=MAX_LEN)
    return embed


def get_embedding(embedding_matrix, USE_GLOVE, VOCAB, EMBED_HIDDEN_SIZE,
               TRAIN_EMBED, MAX_LEN):
    # GLOVE_STORE = 'precomputed_glove.weights'
    if USE_GLOVE:
        embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], input_length=MAX_LEN,
                          trainable=TRAIN_EMBED)
    else:
        embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, input_length=MAX_LEN)
    return embed


def base_model(embed, MAX_LEN, SENT_HIDDEN_SIZE, ACTIVATION, DP, L2,
            LABEL_NUM, OPTIMIZER, MLP_LAYER=3, kind='max'):
    # max, sum, average

    print('Build model...')

    BaseEmbeddings = keras.layers.core.Lambda(lambda x: K.max(x, axis=1), output_shape=(SENT_HIDDEN_SIZE,))
    if kind == 'sum':
        BaseEmbeddings = keras.layers.core.Lambda(lambda x: K.sum(x, axis=1), output_shape=(SENT_HIDDEN_SIZE,))
    elif kind == 'average':
        BaseEmbeddings = keras.layers.core.Lambda(lambda x: K.sum(x, axis=1), output_shape=(SENT_HIDDEN_SIZE,))

    translate = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

    premise = Input(shape=(MAX_LEN,), dtype='int32')

    prem = embed(premise)
    prem = translate(prem)
    prem = BaseEmbeddings(prem)
    prem = BatchNormalization()(prem)

    joint = Dropout(DP)(prem)
    for i in range(MLP_LAYER):
        joint = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, W_regularizer=l2(L2) if L2 else None)(joint)
        joint = Dropout(DP)(joint)
        joint = BatchNormalization()(joint)

    pred = Dense(LABEL_NUM, activation='softmax')(joint)

    model = Model(inputs=premise, outputs=pred)
    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model


def rnn_model(embed, MAX_LEN, SENT_HIDDEN_SIZE, ACTIVATION, DP, L2,
            LABEL_NUM, OPTIMIZER, MLP_LAYER, LAYERS, RNN_Cell='LSTM'):
    """
    :param embed:
    :param VOCAB:
    :param MAX_LEN:
    :param SENT_HIDDEN_SIZE:
    :param ACTIVATION:
    :param DP:
    :param L2:
    :param LABEL_NUM:
    :param OPTIMIZER:
    :param RNN_Cell: LSTM, BiLSTM, GRU, BiGRU
    :param LAYERS:
    :return:
    """
    print('Build model...')

    RNN = recurrent.LSTM
    if RNN_Cell == 'BiLSTM':
        RNN = lambda *args, **kwargs: Bidirectional(recurrent.LSTM(*args, **kwargs))
    elif RNN_Cell == 'GRU':
        RNN = recurrent.GRU
    elif RNN_Cell == 'BiGRU':
        RNN = lambda *args, **kwargs: Bidirectional(recurrent.GRU(*args, **kwargs))

    rnn_kwargs = dict(units=SENT_HIDDEN_SIZE, dropout=DP, recurrent_dropout=DP)

    translate = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

    premise = Input(shape=(MAX_LEN,), dtype='int32')

    prem = embed(premise)

    prem = translate(prem)

    if LAYERS > 1:
        for l in range(LAYERS - 1):
            rnn = RNN(return_sequences=True, **rnn_kwargs)
            prem = BatchNormalization()(rnn(prem))
    rnn = RNN(return_sequences=False, **rnn_kwargs)
    prem = rnn(prem)
    prem = BatchNormalization()(prem)

    joint = Dropout(DP)(prem)
    for i in range(MLP_LAYER):
        joint = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, W_regularizer=l2(L2) if L2 else None)(joint)
        joint = Dropout(DP)(joint)
        joint = BatchNormalization()(joint)

    pred = Dense(LABEL_NUM, activation='softmax')(joint)

    model = Model(inputs=premise, outputs=pred)
    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model


def cnn_model(embed, MAX_LEN, SENT_HIDDEN_SIZE, ACTIVATION, DP, L2,
            LABEL_NUM, NGRAM_FILTERS, MLP_LAYER, NUM_FILTER, OPTIMIZER):
    print('Build model...')

    translate = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))
    premise = Input(shape=(MAX_LEN,), dtype='int32')
    prem = embed(premise)
    prem = translate(prem)
    prem = Dropout(DP)(prem)

    convolutions = []
    i = 0
    for n_gram in NGRAM_FILTERS:
        i += 1
        cur_conv = Convolution1D(name="conv_" + str(n_gram) + '_' + str(i),
                                 filters=NUM_FILTER,
                                 kernel_size=n_gram,
                                 padding='valid',
                                 activation='relu',
                                 strides=1)(prem)
        # pool
        one_max = MaxPooling1D(pool_size=MAX_LEN - n_gram + 1)(cur_conv)
        flattened = Flatten()(one_max)
        convolutions.append(flattened)

    sentence_vector = concatenate(convolutions, name="sentence_vector")  # hang on to this layer!

    for i in range(MLP_LAYER):
        sentence_vector = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, W_regularizer=l2(L2) if L2 else None)(sentence_vector)
        sentence_vector = Dropout(DP)(sentence_vector)
        # sentence_vector = BatchNormalization()(sentence_vector)

    sentence_vector = Dropout(DP)(sentence_vector)
    pred = Dense(LABEL_NUM, activation="sigmoid", name="sentence_prediction")(sentence_vector)

    model = Model(input=premise, output=pred)
    model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy')

    model.summary()
    return model


def rnn_cnn_model(embed, MAX_LEN, SENT_HIDDEN_SIZE, ACTIVATION, DP, L2,
            LABEL_NUM, OPTIMIZER, MLP_LAYER, LAYERS, NGRAM_FILTERS, NUM_FILTER, RNN_Cell='LSTM'):
    print('Build model...')

    RNN = recurrent.LSTM
    if RNN_Cell == 'BiLSTM':
        RNN = lambda *args, **kwargs: Bidirectional(recurrent.LSTM(*args, **kwargs))
    elif RNN_Cell == 'GRU':
        RNN = recurrent.GRU
    elif RNN_Cell == 'BiGRU':
        RNN = lambda *args, **kwargs: Bidirectional(recurrent.GRU(*args, **kwargs))

    rnn_kwargs = dict(units=SENT_HIDDEN_SIZE, dropout=DP, recurrent_dropout=DP)

    translate = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

    premise = Input(shape=(MAX_LEN,), dtype='int32')

    prem = embed(premise)

    prem = translate(prem)

    if LAYERS > 1:
        for l in range(LAYERS - 1):
            rnn = RNN(return_sequences=True, **rnn_kwargs)
            prem = BatchNormalization()(rnn(prem))
    rnn = RNN(return_sequences=True, **rnn_kwargs)
    prem = rnn(prem)
    prem = BatchNormalization()(prem)

    # cnn model
    convolutions = []
    i = 0
    for n_gram in NGRAM_FILTERS:
        i += 1
        cur_conv = Conv1D(name="conv_" + str(n_gram) + '_' + str(i),
                                 filters=NUM_FILTER,
                                 filter_length=n_gram,
                                 border_mode='valid',
                                 activation='relu',
                                 subsample_length=1)(prem)
        # pool
        one_max = MaxPooling1D(pool_length=MAX_LEN - n_gram + 1)(cur_conv)
        flattened = Flatten()(one_max)
        convolutions.append(flattened)

    sentence_vector = concatenate(convolutions, name="sentence_vector")  # hang on to this layer!

    for i in range(MLP_LAYER):
        sentence_vector = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, W_regularizer=l2(L2) if L2 else None)(
            sentence_vector)
        sentence_vector = Dropout(DP)(sentence_vector)
        sentence_vector = BatchNormalization()(sentence_vector)

    sentence_vector = Dropout(DP)(sentence_vector)
    pred = Dense(LABEL_NUM, activation="softmax", name="sentence_prediction")(sentence_vector)

    model = Model(input=premise, output=pred)
    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

def rnn_att_model(embed, MAX_LEN, SENT_HIDDEN_SIZE, ACTIVATION, DP, L2,
                  LABEL_NUM, OPTIMIZER, MLP_LAYER, LAYERS, RNN_Cell='LSTM'):
    print('Build model...')

    RNN = recurrent.LSTM
    if RNN_Cell == 'BiLSTM':
        RNN = lambda *args, **kwargs: Bidirectional(recurrent.LSTM(*args, **kwargs))
    elif RNN_Cell == 'GRU':
        RNN = recurrent.GRU
    elif RNN_Cell == 'BiGRU':
        RNN = lambda *args, **kwargs: Bidirectional(recurrent.GRU(*args, **kwargs))

    rnn_kwargs = dict(units=SENT_HIDDEN_SIZE, dropout=DP, recurrent_dropout=DP)

    translate = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

    premise = Input(shape=(MAX_LEN,), dtype='int32')

    prem = embed(premise)

    # prem = translate(prem)

    if LAYERS > 1:
        for l in range(LAYERS - 1):
            rnn = RNN(return_sequences=True, **rnn_kwargs)
            prem = BatchNormalization()(rnn(prem))
    rnn = RNN(return_sequences=True, **rnn_kwargs)
    prem = rnn(prem)

    prem = Attention(MAX_LEN)(prem)

    joint = Dropout(DP)(prem)
    for i in range(MLP_LAYER):
        joint = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, kernel_regularizer=l2(L2) if L2 else None)(joint)
        joint = Dropout(DP)(joint)
        # joint = BatchNormalization()(joint)

    pred = Dense(LABEL_NUM, activation='softmax')(joint)

    model = Model(inputs=premise, outputs=pred)
    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model
