#!/usr/bin/env python3
#-*-coding:utf-8-*-
# 
# 玉山人工智慧公開挑戰賽2020夏季賽 NLP應用挑戰賽
# 判斷該新聞內文是否含有AML相關焦點人物，並擷取出焦點人物名單
# model api 
#
# @author: brainchild

# package
from flask import Flask, request, jsonify
import datetime
import hashlib
import re
import os
import codecs
import time
import numpy as np
import pandas as pd
from collections import Counter

import keras.callbacks
from keras.models import Model
from keras.layers import Input, Lambda, Bidirectional, LSTM, Dense
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer
from keras_bert import AdamWarmup, calc_train_steps
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

import tensorflow as tf
from tensorflow.python.keras.backend import set_session
# udf funtion
from package.predict_pack import *

app = Flask(__name__)

# log
today = datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d')
log_path = f'./log/log_{today}.csv'
log_init = pd.DataFrame({'esun_uuid':['init'], 'content':['init'], 'pred_name': ['init']})
log_init.to_csv(log_path, index=False)

# error log
error_log_path = f'./log/error_log_{today}.csv'
error_log_init = pd.DataFrame({'esun_uuid':['init'],'error': ['init']})
error_log_init.to_csv(error_log_path, index=False)

# 參數設定
model_path = r'./data'
config_path = os.path.join(model_path, r'bert/bert_config.json')
checkpoint_path = os.path.join(model_path, r'bert/bert_model.ckpt')
dict_path = os.path.join(model_path, r'bert/vocab.txt')
bert_LSTM_model_path = os.path.join(model_path, r'model1.h5')
bert_LSTM_model_path_2 = os.path.join(model_path, r'model1_5.h5')
ner_model_path = os.path.join(model_path, r'ner_model.h5')
aml_model_2_path = os.path.join(model_path, r'model2.h5')

maxlen = 512
maxlen_aml = 512
maxlen_ner = 512
input_shape = (maxlen_ner, )
maxlen_sentences = 256

# 預載模型
gpu_options = tf.GPUOptions(allow_growth=True)
# model 1
sess1 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
graph1 = tf.get_default_graph()
set_session(sess1)
model = bert_LSTM_model()
model.load_weights(bert_LSTM_model_path)

# model 1.5
sess1_5 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
graph1_5 = tf.get_default_graph()
set_session(sess1_5)
model1_5 = bert_model_1_5()
model1_5.load_weights(bert_LSTM_model_path_2)

# model ner
sess2 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
graph2 = tf.get_default_graph()
set_session(sess2)
ner_model = bert_BiLSTM_CRF_model()
ner_model.load_weights(ner_model_path)

# # model 2
sess3 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
graph3 = tf.get_default_graph()
set_session(sess3)
model2 = bert_model()
model2.load_weights(aml_model_2_path)

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'gimy0422@gmail.com'    #
SALT = 'brainchild'                     #
#########################################

def generate_server_uuid(input_string):
    """ Create your own server_uuid
    @param input_string (str): information to be encoded as server_uuid
    @returns server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string+SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid

def predict(esun_uuid, input_data, model, model1_5, ner_model, model2, aml_threshold=0.4, threshold=0.3):

    # phase 1 
    # model 1: 用來預測該文章是否跟AML相關    

    tokenizer, token_dict = create_tokenizer(dict_path=dict_path)
    test_news = clean_marks(input_data)
    test_news_2 = rebuild_sentence(test_news, maxlen)
    data = encoded(tokenizer=tokenizer, data=test_news_2, maxlen=maxlen)

    # 第一階段預測，大於aml_threshold者為疑似aml文章   
    global sess1
    global graph1
    with graph1.as_default():
        set_session(sess1)
        prediction_1 = model.predict(data)

    # 第一階段如果大於 aml threshold, 過model 1.5再判斷
    if prediction_1 >= aml_threshold:

        data_2 = encoded(tokenizer=tokenizer, data=test_news_2, maxlen=maxlen_aml)
        global sess1_5
        global graph1_5
        with graph1_5.as_default():
            set_session(sess1_5)           
            prediction1_5 = model1_5.predict(data_2) 

        # 階段1.5 > 0.3 則進入第二階段
        if prediction1_5 >= 0.3:      
            # phase 2 
            # model ner: 用於提取文章中的人名    
            try:
                test_ner = split_content(test_news)
                name_list = []              
                for i in range(len(test_ner)):
                    input_id, segment_id, mask_input = encoded(tokenizer, test_ner[i], maxlen=maxlen_ner)
                    global sess2
                    global graph2
                    with graph2.as_default():
                        set_session(sess2)
                        ner_prediction = ner_model.predict([input_id, segment_id, mask_input])
                    y_pred = np.argmax(ner_prediction, axis=-1)                  
                    tmp_list = get_name(input_id, y_pred, token_dict)[0]
                    name_list.extend(tmp_list)

                # 處理拿到的人名
                name_list = list(set(name_list))
                name_list = ['' if len(re.findall('[()<>{}\[\]]', n)) > 0 else n for n in name_list]
            except Exception as e:
                err_log = pd.DataFrame({'esun_uuid': [esun_uuid],'error': [e]})
                err_log.to_csv(error_log_path, mode='a', header=False, index=False)
                prediction = []
            # phase 3-1
            # data cleansing & create dataset 

            # 清理人名, 依照原文補回字典沒有的字, 
            # 即將出現 ?或 !的人名補回原形
            try:
                for i, n in enumerate(name_list):
                    if ('?' in n) | ('!' in n):
                        reexp = n.replace('?', '.').replace('!', '.')
                        reexp = re.compile(reexp, re.IGNORECASE)
                        name_list[i] = re.search(reexp, test_news).group() 
                name_list = list(set(name_list))  
                # 取文章內出現人名的前後文        
                sentence_list, innocent_list = create_sentence_list(test_news, name_list)   

                # 填補 innocent_list
                innocent_list = list(set(innocent_list))
                innocent_list = innocent_list_patch(innocent_list, name_list)

                # 準備dataset
                aml_dataset = create_dataset(sentence_list)
            except Exception as e:
                err_log = pd.DataFrame({'esun_uuid': [esun_uuid],'error': [e]})
                err_log.to_csv(error_log_path, mode='a', header=False, index=False)
                prediction = []

            # phase 3-2
            # model 2: 預測該句子是否跟AML相關, 若相關, 則可依照閾值提取關鍵人名      
            try:
                input_id, segment_id, mask_input = encoded_2(tokenizer, aml_dataset, maxlen=maxlen_sentences)
                global sess3
                global graph3
                with graph3.as_default():
                    set_session(sess3)
                    prediction_2 = model2.predict([input_id, segment_id, mask_input])
                aml_dataset['prediction'] = prediction_2
                aml_dataset['prediction'][aml_dataset['name'].isin(innocent_list)] = 0
                aml_dataset['prediction'] = aml_dataset['prediction'].apply(lambda x: 0 if x < threshold else 1)
                aml_dataset = aml_dataset.groupby(['name'])['prediction'].max().reset_index()
                aml_dataset = aml_dataset[aml_dataset['prediction'] == 1]
                aml_name_list = aml_dataset['name'].values.tolist()
                prediction = list(set(aml_name_list))
            except Exception as e:
                err_log = pd.DataFrame({'esun_uuid': [esun_uuid],'error': [e]})
                err_log.to_csv(error_log_path, mode='a', header=False, index=False)
                prediction = []
        # 無關就直接回傳空list 
        else:
            prediction = []
    else:
        prediction = []

    prediction = _check_datatype_to_list(prediction)
    # log
    log = pd.DataFrame({'esun_uuid':[esun_uuid], 'content':[input_data], 'pred_name': [prediction]})
    log.to_csv(log_path, mode='a', header=False, index=False)
    return prediction

def _check_datatype_to_list(prediction):
    """ Check if your prediction is in list type or not. 
        And then convert your prediction to list type or raise error.
        
    @param prediction (list / numpy array / pandas DataFrame): your prediction
    @returns prediction (list): your prediction in list type
    """
    if isinstance(prediction, np.ndarray):
        _check_datatype_to_list(prediction.tolist())
    elif isinstance(prediction, pd.core.frame.DataFrame):
        _check_datatype_to_list(prediction.values)
    elif isinstance(prediction, list):
        return prediction
    raise ValueError('Prediction is not in list type.')

@app.route('/healthcheck', methods=['POST'])
def healthcheck():
    """ API for health check """
    data = request.get_json(force=True)  
    t = datetime.datetime.now()  
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL+ts)
    server_timestamp = t.strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({'esun_uuid': data['esun_uuid'], 'server_uuid': server_uuid, 'captain_email': CAPTAIN_EMAIL, 'server_timestamp': server_timestamp})

@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API """
    data = request.get_json(force=True)  
    esun_timestamp = data['esun_timestamp']  
    t = datetime.datetime.now()  
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL+ts)  
    try:       
        answer = predict(data['esun_uuid'], data['news'], model, model1_5, ner_model, model2, aml_threshold=0.4, threshold=0.4)
    except:
        raise ValueError('Model error.')        
    server_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({'esun_timestamp': data['esun_timestamp'], 'server_uuid': server_uuid, 'answer': answer, 'server_timestamp': server_timestamp, 'esun_uuid': data['esun_uuid']})

if __name__ == "__main__":    
    # production
    app.run(host='0.0.0.0', port=8080, debug=False)
 
    # dev
    # app.run()
    # app.debug = True
