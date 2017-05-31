import os
import sys
import tqdm
import numpy as np
import pandas as pd
np.random.seed(1024)
reload(sys)
sys.setdefaultencoding('utf-8')



from sklearn.preprocessing import OneHotEncoder,LabelBinarizer,MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,Merge,Flatten,Input,merge
from keras.models import Model,Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D,MaxPooling1D,GlobalMaxPooling1D
from keras import metrics


CHAR_EMBEDDING_FILE = '../ieee_zhihu_cup/char_embedding.txt'
WORD_EMBEDDING_FILE = '../ieee_zhihu_cup/word_embedding.txt'
TRAIN_DATA_FILE = '../ieee_zhihu_cup/question_train_set.txt'
TEST_DATA_FILE = '../ieee_zhihu_cup/question_eval_set.txt'
TOPIC_INFO_FILE = '../ieee_zhihu_cup/topic_info.txt'
TRAIN_LABEL_FILE = '../ieee_zhihu_cup/question_topic_train_set.txt'
#MAX_SEQUENCE_LENGTH = 30
MAX_NB_CHARS = 10000             # w2v 11973, vocab 12982
MAX_NB_WORDS = 400000             # w2v 411720, vocab 568825
EMBEDDING_DIM = 256
max_seq_length_title_word=30
max_seq_length_des_word=128


def get_labels():
    f = open(TRAIN_LABEL_FILE)
    idx_list=[]
    y_labels=[]
    for line in f:
        idx,label_list=line.strip().split('\t')
        idx_list.append(int(idx))
        y_labels.append([int(item) for item in label_list.split(',')])
    f.close()


    mlb=MultiLabelBinarizer(sparse_output=True)
    mlb.fit(y_labels)
    NUM_CLASS=len(mlb.classes_)
    y_train=mlb.transform(y_labels)
    pd.to_pickle(y_train,'../input/y.pkl')
    pd.to_pickle(mlb,'../input/mlb_y.pkl')   
 
    y_train_father=pd.read_pickle('../ieee_zhihu_cup/father_labels.pkl')
    mlb_fa=MultiLabelBinarizer(sparse_output=True)
    mlb_fa.fit(y_train_father)
    y_train_father=mlb_fa.transform(y_train_father)
    NUM_CLASS_FA=len(mlb_fa.classes_)
    pd.to_pickle(y_train_father,'../input/y_fa.pkl')
    pd.to_pickle(mlb_fa,'../input/mlb_y_fa.pkl')

    print NUM_CLASS,NUM_CLASS
    
    return y_train,y_train_father

def get_corpus(TYPE='char'):
    texts = []
    term_index= {}
    #data_size = []
    flag=0
    cnt=0
    if TYPE=='char':
        flag=0
    else:
        flag=1
    for fpath in [TRAIN_DATA_FILE,TEST_DATA_FILE]:
	print fpath
        f = open(fpath)
        for line in f:
            if cnt%100000==0:
                print cnt
            terms = line.strip().split('\t')
            if len(terms)==5:
                texts.append(' '.join(terms[1+flag].split(',')))
                texts.append(' '.join(terms[3+flag].split(',')))
            elif len(terms)==3:
                texts.append(' '.join(terms[1+flag].split(',')))
            else:
                continue
            cnt+=1
        f.close()
        #data_size.append(len(texts))
        print len(texts)
    return texts

def get_tokenizer():
    texts_char=get_corpus(TYPE='char')
    tokenizer_char=Tokenizer(nb_words=MAX_NB_CHARS)
    tokenizer_char.fit_on_texts(texts_char)
    pd.to_pickle(tokenizer_char,'../input/tokenizer_char_10000.pkl')
    print 'save token_char_10000'


    texts_word=get_corpus(TYPE='word')
    tokenizer_word=Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer_word.fit_on_texts(texts_word)
    pd.to_pickle(tokenizer_word,'../input/tokenizer_word_400000.pkl')
    print 'save token_word_400000'
    
    return tokenizer_char,tokenizer_word

def pre_data(fpath=TEST_DATA_FILE):
    idx,title_char,title_word,des_char,des_word=[],[],[],[],[]
    f = open(fpath)
    cnt=0
    for line in f:
        if cnt%100000==0:
            print cnt
        terms = line.strip().split('\t')
        idx.append(int(terms[0]))
        if len(terms)==5:
            title_char.append(terms[1])
            title_word.append(terms[2])
            des_char.append(terms[3])
            des_word.append(terms[4])
        elif len(terms)==3:
            title_char.append('')
            title_word.append(terms[1])
            des_char.append('')
            des_word.append(terms[2])
        else:
            title_char.append('')
            title_word.append('')
            des_char.append('')
            des_word.append('')
        cnt+=1
    f.close()
    print 'SIZE:',len(idx)
    return idx,title_char,title_word,des_char,des_word



if __name__ == '__main__':


###################

    tokenizer_char,tokenizer_word=get_tokenizer()
    y_train,y_train_father=get_labels()  


    from gensim.models import KeyedVectors
    word2vec_char = KeyedVectors.load_word2vec_format(CHAR_EMBEDDING_FILE,         binary=False)
    print('Found %s word vectors of word2vec' % len(word2vec_char.vocab))              # 11973
    word2vec_word = KeyedVectors.load_word2vec_format(WORD_EMBEDDING_FILE,         binary=False)
    print('Found %s word vectors of word2vec' % len(word2vec_word.vocab))              # 411720


    # ```python
    print('Preparing embedding matrix')
    nb_words = min(MAX_NB_CHARS, len(tokenizer_char.word_index))+1
    embedding_matrix_char = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in tokenizer_char.word_index.items():
        if word in word2vec_char.vocab and i < nb_words:
            embedding_matrix_char[i] = word2vec_char.word_vec(word)
    print embedding_matrix_char.shape
    pd.to_pickle(embedding_matrix_char,'../input/embed_matrix_char.pkl')            ###
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix_char, axis=1) == 0))
    # ```python


    print('Preparing embedding matrix')
    nb_words = min(MAX_NB_WORDS, len(tokenizer_word.word_index))+1
    embedding_matrix_word = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in tokenizer_word.word_index.items():
        if word in word2vec_word.vocab and i < nb_words:
            embedding_matrix_word[i] = word2vec_word.word_vec(word)
    print embedding_matrix_word.shape
    pd.to_pickle(embedding_matrix_word,'../input/embed_matrix_word.pkl')            ###
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix_word, axis=1) == 0))




    test_idx,_,test_title_word,_,test_des_word=pre_data(TEST_DATA_FILE)
    train_idx,_,train_title_word,_,train_des_word=pre_data(TRAIN_DATA_FILE)

    train_title_word_input=pad_sequences(tokenizer_word.texts_to_sequences(train_title_word),
                                     maxlen=max_seq_length_title_word,padding='post')
    train_des_word_input=pad_sequences(tokenizer_word.texts_to_sequences(train_des_word),
                                     maxlen=max_seq_length_des_word,padding='post')

    test_title_word_input=pad_sequences(tokenizer_word.texts_to_sequences(test_title_word),
                                     maxlen=max_seq_length_title_word,padding='post')
    test_des_word_input=pad_sequences(tokenizer_word.texts_to_sequences(test_des_word),
                                     maxlen=max_seq_length_des_word,padding='post')

    print train_title_word_input.shape,train_des_word_input.shape


    idx=np.arange(y_train.shape[0])
    np.random.seed(1024)
    np.random.shuffle(idx)
    tr_id=idx[:int(y_train.shape[0]*0.9)]
    te_id=idx[int(y_train.shape[0]*0.9):]

    tr_title_word_input=train_title_word_input[tr_id]
    tr_des_word_input=train_des_word_input[tr_id]
    tr_y=y_train[tr_id]
    tr_y_fa=y_train_father[tr_id]

    te_title_word_input=train_title_word_input[te_id]
    te_des_word_input=train_des_word_input[te_id]
    te_y=y_train[te_id]
    te_y_fa=y_train_father[te_id]

    print tr_title_word_input.shape,tr_des_word_input.shape,tr_y.shape,tr_y_fa.shape

    print te_title_word_input.shape,te_des_word_input.shape,te_y.shape,te_y_fa.shape

    pd.to_pickle(tr_title_word_input,'../input/train_title_word.pkl')           ##
    pd.to_pickle(te_title_word_input,'../input/valid_title_word.pkl')	    ##
    pd.to_pickle(tr_des_word_input,'../input/train_des_word.pkl')               ##
    pd.to_pickle(te_des_word_input,'../input/valid_des_word.pkl')               ##

    pd.to_pickle(tr_y,'../input/train_y.pkl')				##
    pd.to_pickle(te_y,'../input/valid_y.pkl')				##
    pd.to_pickle(tr_y_fa,'../input/train_y_fa.pkl')				##
    pd.to_pickle(te_y_fa,'../input/valida_y_fa.pkl')			##



    pd.to_pickle(test_title_word_input,'../input/test_title_word.pkl')       ## 
    pd.to_pickle(test_des_word_input,'../input/test_des_word.pkl')           ##



def get_data(USE_FA = True,USE_GLOVE_EMBED = True):

    PATH='/home/t-zhga/zhihu/input/'
    TRAIN_FILE=['train_title_word.pkl','train_des_word.pkl','train_y.pkl']
    VALID_FILE=['valid_title_word.pkl','valid_des_word.pkl','valid_y.pkl']
    TEST_FILE=['test_title_word.pkl','test_des_word.pkl']
    EMBED_FILE='embed_matrix_word.pkl'
    MLB_FILE='mlb_y.pkl'
    if USE_FA:
	TRAIN_FILE.append('train_y_fa.pkl')
	VALID_FILE.append('valid_y_fa.pkl')


    TRAIN,VALID,TEST=[],[],[]
    for ft,fv in zip(TRAIN_FILE,VALID_FILE):
	TRAIN.append(pd.read_pickle(PATH+ft))
	VALID.append(pd.read_pickle(PATH+fv))
    for fte in TEST_FILE:
	TEST.append(pd.read_pickle(PATH+fte))
    embed_matrix=pd.read_pickle(PATH+EMBED_FILE)
    mlb_y=pd.read_pickle(PATH+MLB_FILE)
    VOCAB=MAX_NB_WORDS+1
    return [TRAIN[:2],TRAIN[2:]],[VALID[:2],VALID[2:]],TEST[:2],embed_matrix,mlb_y,VOCAB
	

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


def batch_generator(train,y,batch_size=128,shuffle=True,use_fa=False):
    sample_size = train[0].shape[0]
    index_array = np.arange(sample_size)
    while 1:
        if shuffle:
            np.random.shuffle(index_array)
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            X_batch_title = train[0][batch_ids]
            X_batch_des = train[-1][batch_ids]
            X_batch = [X_batch_title,X_batch_des]
            y_batch = y[0][batch_ids].toarray()
	    if use_fa:
            	y_batch_fa = y[-1][batch_ids].toarray()
                yield X_batch,[y_batch,y_batch_fa]
	    else:
		yield X_batch,[y_batch]


def batch_generator_v2(train,y,batch_size=128,shuffle=True,use_fa=False):
    sample_size = train[0].shape[0]
    index_array = np.arange(sample_size)
    while 1:
        if shuffle:
            np.random.shuffle(index_array)
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            X_batch_title = train[0][batch_ids]
            X_batch_des = train[-1][batch_ids]
            X_batch = np.hstack([X_batch_title,X_batch_des])
            y_batch = y[0][batch_ids].toarray()
            if use_fa:
                y_batch_fa = y[-1][batch_ids].toarray()
                yield X_batch,[y_batch,y_batch_fa]
            else:
                yield X_batch,[y_batch]
 
