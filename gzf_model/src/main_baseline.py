import os
import sys
import pandas as pd
import numpy as np
from data_loader import get_data,batch_generator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D,MaxPooling1D,GlobalMaxPooling1D
from keras import metrics
from model_zoo import get_embedding, rnn_model,rnn_att_model,base_model
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#config.device='cuda3'
def get_testid():
    f = open('../ieee_zhihu_cup/question_eval_set.txt','r')
    idx_list=[]
    y_labels=[]
    for line in f:
        idx=line.strip().split('\t')[0]
        idx_list.append(int(idx))
        #y_labels.append([int(item) for item in label_list.split(',')])
    f.close()
    return idx_list

def run():
    print('Loading data...')

    #data_folder = '../data/weibo_xiaoice_large'
    training, validation, test, embedding_matrix,  label_map, VOCAB = get_data(USE_FA = False,USE_GLOVE_EMBED = True)
    tr_gen = batch_generator(training[0],training[-1],batch_size=256,shuffle=True)
    te_gen = batch_generator(validation[0],validation[-1],batch_size=1024,shuffle=False)
    
    print('VOCAB size:{}'.format(VOCAB))

    # Summation of word embeddings
    LAYERS = 1
    USE_GLOVE = False
    TRAIN_EMBED = True
    EMBED_HIDDEN_SIZE = 128
    SENT_HIDDEN_SIZE = 256
    BATCH_SIZE = 512
    PATIENCE = 6  # 8
    MAX_EPOCHS = 100
    MAX_LEN_TITLE = 30
    MAX_LEN_DES = 128
    DP = 0.2
    L2 = 4e-06
    ACTIVATION = 'relu'
   # OPTIMIZER = 'rmsprop'
    OPTIMIZER = 'adam'
    MLP_LAYER = 0
    NGRAM_FILTERS = [1, 2, 3, 4]
    NUM_FILTER = 128
    RNN_Cell = 'BiLSTM'

    print('Embed / Sent = {}, {}'.format(EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE))
    print('GloVe / Trainable Word Embeddings = {}, {}'.format(USE_GLOVE, TRAIN_EMBED))

    LABEL_NUM = len(label_map.classes_)


    bst_model_path = '../model/base_model_v1.hdf5'
    pred_path      = '../res/base_model_v1.pkl'
    res_path       = '../res/base_model_v1.csv'

    embed= get_embedding(embedding_matrix, USE_GLOVE, VOCAB, EMBED_HIDDEN_SIZE,
               TRAIN_EMBED, None)


    model = base_model(embed, MAX_LEN_TITLE ,MAX_LEN_DES, SENT_HIDDEN_SIZE, ACTIVATION, DP, L2,
            LABEL_NUM, OPTIMIZER, MLP_LAYER=1, kind='average')
 
    early_stopping =EarlyStopping(monitor='val_loss', patience=4)    
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    #print training[0][0].shape[0],validation[0][0].shape[0]
    #model.load_weights(bst_model_path)    
    model.fit_generator(tr_gen,steps_per_epoch=int(training[0][0].shape[0]/BATCH_SIZE)+1,
                    epochs=100,verbose=1,
                    validation_data=te_gen,
                    validation_steps=int(validation[0][0].shape[0]/BATCH_SIZE)+1,max_q_size=20,
                    callbacks=[early_stopping,model_checkpoint])
    
    print 'load weights'
    model.load_weights(bst_model_path)
    pred=model.predict(test)
    pd.to_pickle(pred,pred_path)

    def get_ans(pred=pred,idd=np.random.random(3000)):
        pred=pred.argsort(axis=1)[:,::-1][:,:5]
        ll=label_map.classes_
        ans=[[ll[item] for item in items] for items in pred]
        res=pd.DataFrame(ans)
        res.index=idd
        return res
    test_idx=get_testid()
    ans=get_ans(pred,test_idx)
    ans.to_csv(res_path,index=True,header=False)
    
if __name__ == '__main__':
    run()
