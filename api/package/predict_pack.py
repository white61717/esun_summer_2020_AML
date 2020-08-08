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
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras_bert import Tokenizer
from keras_bert import AdamWarmup, calc_train_steps

# 參數區
model_path = r'./data'
config_path = os.path.join(model_path, r'bert/bert_config.json')
checkpoint_path = os.path.join(model_path, r'bert/bert_model.ckpt')
dict_path = os.path.join(model_path, r'bert/vocab.txt')
bert_LSTM_model_path = os.path.join(model_path, r'model1.h5')
ner_model_path = os.path.join(model_path, r'ner_model.h5')
aml_model_2_path = os.path.join(model_path, r'model2.h5')
maxlen = 512
maxlen_aml = 512
maxlen_ner = 512
input_shape = (maxlen_ner, )
maxlen_sentences = 256

# create model
# model 1
def bert_LSTM_model():
    model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True, seq_len=maxlen)
    sequence_output = model.layers[-9].output
    sequence_output = Bidirectional(LSTM(128, return_sequences=False))(sequence_output)
    output = Dense(1, activation='sigmoid')(sequence_output)
    model = Model(model.input, output)
    
    for layer in model.layers:
        layer.trainable = False
    model.layers[-1].trainable = True
    model.layers[-2].trainable = True
    return model

# model 1.5
def bert_model_1_5():
    model1_5 = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True, seq_len=maxlen_aml)
    sequence_output = model1_5.layers[-6].output
    sequence_output = Dense(64, activation='relu')(sequence_output)
    output = Dense(1, activation='sigmoid')(sequence_output)
    model1_5 = Model(model1_5.input, output)
    return model1_5

# ner model
def bert_BiLSTM_CRF_model():
    ner_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True, seq_len=maxlen_ner)
    bert_output = ner_model.layers[-9].output
    X = Lambda(lambda x: x[:, 0: input_shape[0]])(bert_output)
    X = Bidirectional(LSTM(128, return_sequences=True))(X)
    output = CRF(3, sparse_target = True)(X)    
    ner_model = Model(ner_model.input, output)
    
    for layer in ner_model.layers:
        layer.trainable = False
    ner_model.layers[-1].trainable = True
    ner_model.layers[-2].trainable = True
    
    return ner_model
    
# model 2
def bert_model():
    model2 = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True, seq_len=maxlen_sentences)
    sequence_output = model2.layers[-6].output
    sequence_output = Dense(64, activation='relu')(sequence_output)
    output = Dense(1, activation='sigmoid')(sequence_output)
    model2 = Model(model2.input, output)
    return model2

# data cleansing
def clean_marks(content):
    content = re.sub('<[^>]*>|【[^】]*】|（[^）]*）|〔[^〕]*〕', '', content)
    content = content.strip() \
                     .replace('記者', '＜') \
                     .replace('報導', '＞') \
                     .replace('▲', '') \
                     .replace('。　', '。') \
                     .replace('', '') \
                     .replace('.', '') \
                     .replace(' ', '') \
                     .replace('“', '「') \
                     .replace('”', '」')
    content = re.sub('＜[^＞]*＞', '', content)
    content = re.sub('「.{1,4}」', '', content)
    content = re.sub('｜.', '。', content)

    return content

# data encoded 
def create_tokenizer(dict_path):
    
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    tokenizer = Tokenizer(token_dict)
    
    return tokenizer, token_dict

def encoded(tokenizer, data, maxlen):
    
    x, y, z = [], [], []
    x1, x2 = tokenizer.encode(data, max_len=maxlen)
    x3 = list(map(lambda x: 1 if x != 0 else 0, x1))
    x.append(x1)
    y.append(x2)
    z.append(x3)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    data = [x, y, z]

    return data

def encoded_2(tokenizer, data, maxlen):

    x, y, z = [], [], []
    for content in data['sentence']:
        x1, x2 = tokenizer.encode(content, max_len=maxlen)
        x3 = list(map(lambda x: 1 if x != 0 else 0, x1))
        x.append(x1)
        y.append(x2)
        z.append(x3)

    return x, y, z

def rebuild_sentence(content, maxlen):
    
    if len(content) > maxlen:
        return content[:round(maxlen/2)-1]+ '。' + content[len(content) - (maxlen - round(maxlen/2))+2:]
    else:
        return content

def split_content(content):

    if (len(content) > 512) & (len(content) <= 1024):

        s_split = [(i, abs(len(content)//2 - content.find(x)), x) for i, x in enumerate(content.split('。'))]
        idx_left = min(s_split, key=lambda x: x[1])[0]
        first = "。".join([s_split[i][2] for i in range(idx_left)])
        second = "。".join([s_split[i][2] for i in range(idx_left, len(s_split))])    
        contents = [first, second]
        
        return contents

    elif len(content) > 1024:

        s_split1 = [(i, abs(len(content)//3 - content.find(x)), x) for i, x in enumerate(content.split('。'))]
        s_split2 = [(i, abs(len(content)*2//3 - content.find(x)), x) for i, x in enumerate(content.split('。'))]
        idx_left1 = min(s_split1, key=lambda x: x[1])[0]
        idx_left2 = min(s_split2, key=lambda x: x[1])[0]
        first = "。".join([s_split1[i][2] for i in range(idx_left1)])
        second = "。".join([s_split1[i][2] for i in range(idx_left1, idx_left2)])
        third = "。".join([s_split1[i][2] for i in range(idx_left2, len(s_split1))])
        contents = [first, second, third]
        
        return contents
    
    else:
        return [content]

# feature engineering
# da pattern
def search_da(name: str, sentence: str) -> bool:
    '''
    搜索pattern: "(職業)(人名)(指定動作)" 
    可允許兩人姓名中間有、，再多就不行了。
    '''
    da_list = ['說', '調查', '辦理', '偵訊', '訊問', '諭令', '指揮', '提起', '指出', '表示', 
               '搜索', '認定', '認為', '獲報', '接獲', '依據', '報告', '拿出', '負責']
    if re.search(f'(檢察官|員警|警察|律師|監委|廳長)(...、)?{name}(、...)?則?({"|".join(da_list)})', sentence):
        return True
    else:
        return False

# create target's features
def create_sentence_list(test_news:str, name_list:list) -> list:
    """@test_news: 原文
       @name_list: ner抓出的人名
       @return: sentence_list, 用於下一階段      
    """
    sentence_list = []
    innocent_list = []
    # 切分句子
    test_split_content = test_news
    test_split_content = test_split_content.replace('。','=。').replace('，','*，').replace('？','+？').replace('；','{；')
    test_split_content = re.split('，|。|？|；', test_split_content)
    test_split_content = list(map(lambda x: x.replace('=','。').replace('*','，').replace('+','？').replace('{','；'), test_split_content))
    test_split_content = list(filter(None, test_split_content))
    # 原人名list排序 
    name_list.sort(key=lambda x: len(x), reverse=True)
    for name in name_list:
        tmp_name_list = name_list.copy()
        tmp_name_list.remove(name)
        full_name_list = tmp_name_list.copy()
        tmp_name_list = [n for n in tmp_name_list if ((len(n) < 3) & (n[0] != name[0])) | (len(n) > 2)]
        tmp_name_list = [n.replace('?', '\?') for n in tmp_name_list]
        # 建立sentences list, 即上拼接下文
        for i, s in enumerate(test_split_content):
            sentence = ''
            if name in s:
                # 正常有前後文共3句的情況
                # 第1句出現句號 -> 拼2+3
                # 第2句出現句號 -> 拼1+2
                # 第3句或沒有句號 -> 拚1+2+3
                # 若名字出現在頭尾的例外處理                      
                try:
                    if i != 0:
                        start = test_split_content[i-1]
                        mid = test_split_content[i]
                        end = test_split_content[i+1]
                        
                        if start.find('。') > 0:
                            start = ''
                        if mid.find('。') > 0:
                            end = ''
                        if sum([1 for n in full_name_list if n in start]) > 0:
                            start = ''
                        if sum([1 for n in full_name_list if n in end]) > 0:
                            end = ''
                        mid = re.sub('|'.join(tmp_name_list), '', mid)            
                        sentence = start + mid + end     
                    else:
                        mid = test_split_content[i]
                        end = test_split_content[i+1]
                        if mid.find('。') > 0:
                            end = ''
                        if sum([1 for n in full_name_list if n in end]) > 0:
                            end = ''
                        mid = re.sub('|'.join(tmp_name_list), '', mid)      
                        sentence = mid + end
                except:
                    start = test_split_content[i-1] 
                    mid = test_split_content[i]
                    if start.find('。') > 0:
                        start = ''
                    if sum([1 for n in full_name_list if n in start]) > 0:
                        start = ''
                    mid = re.sub('|'.join(tmp_name_list), '', mid) 
                    sentence = start + mid           
                sentence_list.append((sentence, name))
                # 無罪規則
                if (('無罪定讞' in mid) | ('確定無罪' in mid) | ('無罪確定' in mid)| ('罪嫌不足' in mid) | ('罪證不足' in mid) | ('不起訴' in mid)) & ('、' not in mid):
                    innocent_list.append(name)
                try:
                    if (('無罪定讞' in end) | ('確定無罪' in end) | ('無罪確定' in end)| ('罪嫌不足' in end) | ('罪證不足' in end) | ('不起訴' in end)) & ('、' not in end):
                        innocent_list.append(name)
                except:
                    pass
                # 檢察官規則
                try:
                    if search_da(name, mid+end):
                        innocent_list.append(name)
                except:
                    pass
                
    return sentence_list, innocent_list

# create innocent people list 
def innocent_list_patch(innocent_list:list, name_list:list):

    innocent_name_list = []
    full_name = innocent_list.copy()
    full_3name = list(set([name for name in name_list if len(name) == 3]))    
    a = Counter([name[0] for name in full_3name])
    keep = [k for k,v in a.items() if v == 1]
    
    full_3name_filter = [name for name in full_3name if name[0] in keep]
    name_dict = dict((name[0], name) for name in full_3name_filter)   # ex: {'陳' : '陳水扁'}
    for name in full_name:
        if (name[0] in name_dict.keys()) & (len(name) == 1):
            innocent_name_list.append(name_dict.get(name[0]))
        else:
            innocent_name_list.append(name)
                
    return innocent_name_list    

# create dataset for model 2 
def create_dataset(sentence_list:list) -> pd.DataFrame:
    AML = pd.DataFrame(sentence_list, columns=['sentence', 'name'])     
    name_list = []

    #姓氏表
    first_name = ['浦', '藺', '俄', '嵇', '莘', '吉', '廉', '空', '章', '金', '蔡', '謝', '邴', '任', '江', '雍', '宮', '洪', '範', '闕', '歐', '經', '溫', '吳', 
                  '匡', '巫', '薄', '尚', '武', '全', '龔', '陽', '蒲', '錢', '關', '戈', '慎', '朱', '施', '刁', '文', '養', '桑', '閆', '汝', '談', '能', '蓋', 
                  '毛', '厙', '白', '王', '郁', '屠', '東', '杜', '靳', '涂', '笪', '欒', '郎', '扶', '晏', '封', '倪', '艾', '冷', '於', '祿', '陳', '陶', '宦', 
                  '盧', '沈', '鄧', '聞', '翟', '都', '苗', '戎', '咸', '米', '弘', '池', '穆', '仲', '林', '童', '牛', '蓬', '何', '翁', '佘', '幸', '路', '蕭', 
                  '班', '李', '樊', '壽', '易', '支', '安', '費', '畢', '俞', '祖', '酆', '羿', '查', '牧', '齊', '孫', '車', '和', '庾', '黨', '別', '常', '邵', 
                  '有', '汲', '況', '薊', '丁', '皮', '鄔', '曹', '穀', '訾', '卻', '管', '鄢', '仇', '諸', '賈', '湯', '仰', '緱', '瞿', '嚴', '充', '燕', '甯', 
                  '索', '薛', '勾', '商', '廖', '申', '左', '韶', '阮', '璩', '谷', '黎', '陰', '晁', '相', '狄', '鞏', '從', '愛', '鮑', '夔', '寇', '汪', '利',
                  '范', '彭', '暴', '逯', '惠', '蒼', '姬', '衛', '遊', '邱', '荊', '國', '蔣', '屈', '焦', '康', '豐', '于', '邢', '程', '周', '呂', '平', '帥', 
                  '亢', '郝', '凌', '喻', '鐘', '桓', '魯', '步', '盛', '貢', '松', '符', '賴', '芮', '佴', '秦', '越', '佟', '榮', '龐', '暨', '蒯', '楊', '農', 
                  '邰', '宋', '石', '伏', '塗', '危', '貝', '井', '賁', '習', '鈄', '昝', '秋', '奚', '溥', '解', '梅', '徐', '辛', '戚', '段', '乜', '葛', '沙', 
                  '湛', '麻', '馮', '柴', '烏', '席', '幹', '琴', '曾', '干', '賀', '勞', '趙', '韓', '曆', '滕', '郤', '譚', '糜', '沃', '年', '濮', '水', '顏', 
                  '滿', '劉', '虞', '茅', '莊', '蒙', '強', '敖', '顧', '方', '時', '山', '牟', '袁', '饒', '毋', '逄', '融', '單', '潘', '喬', '衡', '鞠', '宰', 
                  '鹹', '郟', '閔', '桂', '史', '董', '盍', '容', '卜', '元', '蔚', '尹', '霍', '黃', '蘇', '冉', '哈', '姜', '廣', '華', '梁', '熊', '逮', '欽', 
                  '隗', '郗', '胡', '殳', '闞', '家', '紀', '伊', '許', '雲', '邊', '終', '益', '昌', '居', '夏', '聶', '楚', '岳', '師', '竇', '司', '龍', '項', 
                  '羅', '甄', '古', '殷', '游', '法', '宓', '麴', '詹', '田', '連', '道', '高', '孔', '花', '馬', '姚', '杭', '簡', '宿', '甘', '柏', '鄭', '冀', 
                  '公', '岑', '儲', '滑', '后', '魚', '郜', '耿', '祁', '郭', '駱', '崔', '葉', '淩', '向', '巢', '弓', '裘', '宣', '鍾', '隆', '富', '雷', '宗', 
                  '侯', '臧', '通', '須', '堵', '晉', '竺', '顔', '印', '傅', '景', '權', '厲', '韋', '包', '樂', '繆', '荀', '成', '孟', '唐', '籍', '紅', '萬', 
                  '尤', '譙', '柯', '婁', '束', '茹', '房', '酈', '墨', '柳', '張', '裴', '雙', '鳳', '懷', '陸', '巴', '莫', '禹', '後', '扈', '羊', '藍', '祝', 
                  '蔔', '鄒', '計', '慕', '閻', '應', '余', '舒', '僪', '那', '伍', '鬱', '胥', '魏', '明', '戴', '餘', '卓', '褚', '鈕', '鄂', '季', '卞']

    # name 拿出來
    full_name = list(AML['name'])
    # 三字人名取 unique
    # full_3name = list(set([n for n in list(set(full_name)) if len(n) == 3]))
    full_3name = list(set([n for n in list(set(full_name)) if len(n) in [3, 4]]))
    full_longname = list(set([n for n in list(set(full_name)) if len(n) > 3]))

    a = Counter([name[0] for name in full_3name])
    keep = [k for k,v in a.items() if v == 1]
    
    name_dict_2 = dict(zip([name[0:2] for name in full_3name], full_3name))  # ex: {'王音': '王音之'}
    name_dict_3 = dict(zip([name[1:] for name in full_longname], full_longname))  
    name_dict_4 = dict(zip([name[:2] for name in full_longname], full_longname)) 
    for name in full_name:
        if (name in name_dict_2.keys()) & (len(name) == 2):
            name_list.append(name_dict_2.get(name))
        elif (name in name_dict_3.keys()):
            name_list.append(name_dict_3.get(name))
        elif (name in name_dict_4.keys()) & (len(name) == 2):
            name_list.append(name_dict_4.get(name))
        else:
            name_list.append(name)
    
    # 排除重複資料、排除一字、兩字簡稱、兩字三字四字姓不在姓氏表中的人
    AML['name'] = name_list
    AML = AML.drop_duplicates()
    AML = AML[AML['name'].apply(lambda x: (len(x) > 1) )]
    AML = AML[~AML['name'].apply(lambda x: (len(x) == 2) & (x[1] in ['男', '嫌', '婦', '夫', '某', '女', '妻', '員', '稱', '家', '哥', '媽', '生',  '揆', '董', '母', '公', '少', '翁', '粉', '仔', '氏', '父', '童', '弟', '嬤', '姐', '姊', '警']))]
    AML = AML[~AML['name'].apply(lambda x: (len(x) == 2) & (x[0] in ['小', '阿', '老']))]
    AML = AML[~AML['name'].apply(lambda x: (len(x) == 2) & (x[0] == x[1]))]
    AML = AML[AML['name'].apply(lambda x: (len(x) > 2) | ((len(x) < 3) & (x[0] in first_name)))]
    AML = AML[~AML['name'].apply(lambda x: (x[0] not in first_name) & (len(x) in (4,3,2)) )]
    # AML = AML[~AML.apply(lambda x: search_da(x['name'], x['sentence']), axis=1)]

    return AML

# predict function
# generally function
def predict_aml(model, data, aml_threshold):
    
    #第一階段預測，大於aml_threshold者為疑似aml文章   
    prediction = model.predict(data)
    prediction[prediction >= aml_threshold] = 1
    prediction[prediction < aml_threshold] = 0
    
    return prediction

# 取得名字 (預測結果為onehot的狀態)
def get_name(input_id, y_pred, token_dict):
    
    label_list = []
    word_dict = {v: k for k, v in token_dict.items()}
    
    for input_data, y in zip(input_id, y_pred):
        people_index = ''.join([str(a) for a in list(y)])
        j = 0
        name_list = []
        split_index = re.findall('[12]2*', people_index)
        name = ''.join([word_dict.get(input_data[index]) for index, value in enumerate(y) if value != 0])
        
        # [UNK], [PAD]會被算成 5 個字元，避免轉換成文字的index因長度不同對不上，故用 1 個字元的其他符號替代
        # 王春甡 -> 王春[UNK] -> 王春?
        name = name.replace('[UNK]','?')
        name = name.replace('[PAD]','!')
        
        for i in split_index:
            name_list.append(name[0+j:len(i)+j])
            j = len(i) + j
            
        name_list = [name for name in name_list]
        label_list.append(list(set(name_list)))
    
    return label_list
