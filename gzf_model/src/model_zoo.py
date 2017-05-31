import os

import keras
import keras.backend as K
import numpy as np
from keras.layers import recurrent, Dense, Input, Dropout, TimeDistributed, Flatten, concatenate
from keras.layers import GlobalMaxPooling1D
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Conv1D
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.regularizers import l2
from keras import metrics

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


def base_model(embed, MAX_LEN_TITLE ,MAX_LEN_DES, SENT_HIDDEN_SIZE, ACTIVATION, DP, L2,
            LABEL_NUM, OPTIMIZER, MLP_LAYER=3, kind='max'):
    # max, sum, average

    print('Build model...')

    BaseEmbeddings = keras.layers.core.Lambda(lambda x: K.max(x, axis=1), output_shape=(SENT_HIDDEN_SIZE,))
    if kind == 'sum':
        BaseEmbeddings = keras.layers.core.Lambda(lambda x: K.sum(x, axis=1), output_shape=(SENT_HIDDEN_SIZE,))
    elif kind == 'average':
        BaseEmbeddings = keras.layers.core.Lambda(lambda x: K.sum(x, axis=1), output_shape=(SENT_HIDDEN_SIZE,))

    translate_title = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))
    premise_title = Input(shape=(MAX_LEN_TITLE,), dtype='int32')
    translate_des = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))
    premise_des = Input(shape=(MAX_LEN_DES,),dtype='int32')

    
    prem_title = embed(premise_title)
    prem_title = translate_title(prem_title)
    prem_title = BaseEmbeddings(prem_title)
    prem_title = BatchNormalization()(prem_title)


    prem_des = embed(premise_des)
    prem_des = translate_des(prem_des)
    prem_des = BaseEmbeddings(prem_des)
    prem_des = BatchNormalization()(prem_des)

    prem =  concatenate([prem_title,prem_des], name="sentence_vector")

    joint = Dropout(DP)(prem)
    for i in range(MLP_LAYER):
        joint = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, W_regularizer=l2(L2) if L2 else None)(joint)
        joint = Dropout(DP)(joint)
        joint = BatchNormalization()(joint)

    pred = Dense(LABEL_NUM, activation='sigmoid')(joint)

    model = Model(input=[premise_title,premise_des], output=pred)
    model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy')

    model.summary()
    return model


def rnn_model(embed, MAX_LEN_TITLE, MAX_LEN_DES, SENT_HIDDEN_SIZE, ACTIVATION, DP, L2,
            LABEL_NUM, OPTIMIZER, MLP_LAYER, LAYERS, RNN_Cell='LSTM'):
    """
    :param embed_title,embed_des:
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

    rnn_kwargs = dict(output_dim=SENT_HIDDEN_SIZE, dropout_W=DP, dropout_U=DP)

    translate_title = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))
    translate_des = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

    premise_title = Input(shape=(MAX_LEN_TITLE,), dtype='int32')
    prem_title = embed(premise_title)
    prem_title = translate_title(prem_title)

    

    premise_des = Input(shape=(MAX_LEN_DES,), dtype='int32')
    prem_des = embed(premise_des)
    prem_des = translate_des(prem_des)

    if LAYERS > 1:
        for l in range(LAYERS - 1):
            rnn_title = RNN(return_sequences=True, **rnn_kwargs)
            prem_title = BatchNormalization()(rnn_title(prem_title))
	    
	    rnn_des = RNN(return_sequences=True, **rnn_kwargs)
	    prem_des = BatchNormalization()(rnn_des(prem_des))
	
    rnn_title = RNN(return_sequences=False, **rnn_kwargs)
    prem_title = rnn_title(prem_title)
    prem_title = BatchNormalization()(prem_title)

    rnn_des = RNN(return_sequences=False, **rnn_kwargs)
    prem_des = rnn_des(prem_des)
    prem_des = BatchNormalization()(prem_des)

    prem =  concatenate([prem_title,prem_des], name="sentence_vector")    

    joint = Dropout(DP)(prem)
    for i in range(MLP_LAYER):
        joint = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, W_regularizer=l2(L2) if L2 else None)(joint)
        joint = Dropout(DP)(joint)
        joint = BatchNormalization()(joint)

    pred = Dense(LABEL_NUM, activation='sigmoid')(joint)

    model = Model(input=[premise_title,premise_des], output=pred)
    model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=[metrics.top_k_categorical_accuracy])

    model.summary()
    return model


def cnn_model(embed, MAX_LEN_TITLE, MAX_LEN_DES, SENT_HIDDEN_SIZE, ACTIVATION, DP, L2,
            LABEL_NUM, NGRAM_FILTERS, MLP_LAYER, NUM_FILTER, OPTIMIZER):
    print('Build model...')

    translate_title = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))
    translate_des = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

    premise_title = Input(shape=(MAX_LEN_TITLE,), dtype='int32')
    prem_title = embed(premise_title)
    prem_title = translate_title(prem_title)



    premise_des = Input(shape=(MAX_LEN_DES,), dtype='int32')
    prem_des = embed(premise_des)
    prem_des = translate_des(prem_des)
    
    convolutions = []
    i = 0
    for n_gram in NGRAM_FILTERS:
        i += 1
        cur_conv_title = Conv1D(
                                 name="conv_title" + str(n_gram) + '_' + str(i),
                                 filters=NUM_FILTER,
                                 kernel_size=n_gram,
                                 padding='valid',
                                 activation='relu',
				 strides=1)(prem_title)
        # pool
        one_max = GlobalMaxPooling1D()(cur_conv_title)
        #flattened = Flatten()(one_max)
        convolutions.append(one_max)

    i = 0
    for n_gram in NGRAM_FILTERS:
        i += 1
        cur_conv_des = Conv1D(
                                 name="conv_des" + str(n_gram) + '_' + str(i),
                                 filters=NUM_FILTER,
                                 kernel_size=n_gram,
                                 padding='valid',
                                 activation='relu',
				 strides=1)(prem_des)
        # pool
        one_max = GlobalMaxPooling1D()(cur_conv_des)
        #flattened = Flatten()(one_max)
        convolutions.append(one_max)

    sentence_vector = concatenate(convolutions, name="sentence_vector")  # hang on to this layer!

    for i in range(MLP_LAYER):
        sentence_vector = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION,  kernel_regularizer=l2(L2) if L2 else None)(sentence_vector)
        sentence_vector = Dropout(DP)(sentence_vector)
        sentence_vector = BatchNormalization()(sentence_vector)

    sentence_vector = Dropout(DP)(sentence_vector)
    pred = Dense(LABEL_NUM, activation="sigmoid", name="sentence_prediction")(sentence_vector)

    model = Model(input=[premise_title,premise_des], output=pred)
    model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy')

    model.summary()
    return model




def rnn_cnn_model(embed, MAX_LEN_TITLE, MAX_LEN_DES, SENT_HIDDEN_SIZE, ACTIVATION, DP, L2,
            LABEL_NUM, OPTIMIZER, MLP_LAYER, LAYERS, NGRAM_FILTERS, NUM_FILTER, RNN_Cell='LSTM'):
    print('Build model...')

    RNN = recurrent.LSTM
    if RNN_Cell == 'BiLSTM':
        RNN = lambda *args, **kwargs: Bidirectional(recurrent.LSTM(*args, **kwargs))
    elif RNN_Cell == 'GRU':
        RNN = recurrent.GRU
    elif RNN_Cell == 'BiGRU':
        RNN = lambda *args, **kwargs: Bidirectional(recurrent.GRU(*args, **kwargs))
    

    rnn_kwargs = dict(output_dim=SENT_HIDDEN_SIZE, dropout_W=DP, dropout_U=DP)

    translate_title = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))
    translate_des = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

    premise_title = Input(shape=(MAX_LEN_TITLE,), dtype='int32')
    prem_title = embed(premise_title)
    prem_title = translate_title(prem_title)



    premise_des = Input(shape=(MAX_LEN_DES,), dtype='int32')
    prem_des = embed(premise_des)
    prem_des = translate_des(prem_des)

    if LAYERS > 1:
        for l in range(LAYERS - 1):
            rnn_title = RNN(return_sequences=True, **rnn_kwargs)
            prem_title = BatchNormalization()(rnn_title(prem_title))

            rnn_des = RNN(return_sequences=True, **rnn_kwargs)
            prem_des = BatchNormalization()(rnn_des(prem_des))

    rnn_title = RNN(return_sequences=True, **rnn_kwargs)
    prem_title = rnn_title(prem_title)
    prem_title = BatchNormalization()(prem_title)

    rnn_des = RNN(return_sequences=True, **rnn_kwargs)
    prem_des = rnn_des(prem_des)
    prem_des = BatchNormalization()(prem_des)

    #prem =  merge([prem_title,prem_des], name="sentence_vector",mode = 'concat')

    # cnn model
    convolutions = []
    i = 0
    for n_gram in NGRAM_FILTERS:
        i += 1
        cur_conv_title = Conv1D(
				 name="conv_title" + str(n_gram) + '_' + str(i),
				 nb_filter=NUM_FILTER,
                                 filter_length=n_gram,
                                 border_mode='valid',
                                 activation='relu',
				 subsample_length=1)(prem_title)
        # pool
        one_max = GlobalMaxPooling1D()(cur_conv_title)
        #flattened = Flatten()(one_max)
        convolutions.append(one_max)

    i = 0
    for n_gram in NGRAM_FILTERS:
        i += 1
        cur_conv_des = Conv1D(
				 name="conv_des" + str(n_gram) + '_' + str(i),
                                 nb_filter=NUM_FILTER,
                                 filter_length=n_gram,
                                 border_mode='valid',
                                 activation='relu',
				 subsample_length=1
                                 )(prem_des)
        # pool
        one_max = GlobalMaxPooling1D()(cur_conv_des)
        #flattened = Flatten()(one_max)
        convolutions.append(one_max)


    sentence_vector = concatenate(convolutions, name="sentence_vector")  # hang on to this layer!


    for i in range(MLP_LAYER):
        sentence_vector = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, W_regularizer=l2(L2) if L2 else None)(
            sentence_vector)
        sentence_vector = Dropout(DP)(sentence_vector)
        sentence_vector = BatchNormalization()(sentence_vector)

    sentence_vector = Dropout(DP)(sentence_vector)
    pred = Dense(LABEL_NUM, activation="sigmoid", name="sentence_prediction")(sentence_vector)

    model = Model(input=[premise_title,premise_des], output=pred)
    model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=[metrics.top_k_categorical_accuracy])

    model.summary()
    return model

def rnn_att_model(embed, MAX_LEN_TITLE, MAX_LEN_DES, SENT_HIDDEN_SIZE, ACTIVATION, DP, L2,
                  LABEL_NUM, OPTIMIZER, MLP_LAYER, LAYERS, RNN_Cell='LSTM'):
    print('Build model...')

    RNN = recurrent.LSTM
    if RNN_Cell == 'BiLSTM':
        RNN = lambda *args, **kwargs: Bidirectional(recurrent.LSTM(*args, **kwargs))
    elif RNN_Cell == 'GRU':
        RNN = recurrent.GRU
    elif RNN_Cell == 'BiGRU':
        RNN = lambda *args, **kwargs: Bidirectional(recurrent.GRU(*args, **kwargs))

    rnn_kwargs = dict(units=SENT_HIDDEN_SIZE, recurrent_dropout=DP, dropout=DP)

    translate_title = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))
    translate_des = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

    premise_title = Input(shape=(MAX_LEN_TITLE,), dtype='int32')

    prem_title = embed(premise_title)

    prem_title = translate_title(prem_title)

    
    premise_des = Input(shape=(MAX_LEN_DES,), dtype='int32')

    prem_des = embed(premise_des)

    prem_des = translate_des(prem_des)

    
    if LAYERS > 1:
        for l in range(LAYERS - 1):
            rnn_title = RNN(return_sequences=True, **rnn_kwargs)
            prem_title = BatchNormalization()(rnn(prem_title))
	    rnn_des = RNN(return_sequences=True, **rnn_kwargs)
	    prem_des = BatchNormalization()(rnn(prem_des))

    rnn_title = RNN(return_sequences=True, **rnn_kwargs)
    prem_title = rnn_title(prem_title)

    rnn_des = RNN(return_sequences=True, **rnn_kwargs)
    prem_des = rnn_des(prem_des)

    prem_title = Attention(MAX_LEN_TITLE)(prem_title)
    prem_des = Attention(MAX_LEN_DES)(prem_des)
    
    joint_title = Dropout(DP)(prem_title)
    joint_des = Dropout(DP)(prem_des)
   
    joint = concatenate([joint_title,joint_des], name="sentence_vector") 
    for i in range(MLP_LAYER):
        joint = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, kernel_regularizer=l2(L2) if L2 else None)(joint)
        joint = Dropout(DP)(joint)
        # joint = BatchNormalization()(joint)

    pred = Dense(LABEL_NUM, activation='sigmoid')(joint)

    model = Model(inputs=[premise_title,premise_des], outputs=pred)
    model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=[metrics.top_k_categorical_accuracy])

    model.summary()
    return model
