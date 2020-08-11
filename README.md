# 玉山人工智慧公開挑戰賽2020夏季賽 - NLP應用挑戰賽
* by Brainchild
## 預測說明
* 判斷該新聞內文是否含有AML相關焦點人物，並擷取出焦點人物名單
## 文件說明
* create_model.ipynb - 製作model
* api - 使用flask 將model佈署至GCP
* docs - 相關說明文件
## pre-train model
* [BERT](https://github.com/google-research/bert)
    - [BERT-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip): Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters
## 摘要
用Bert分別訓練四階段模型
1.	犯罪模型（Bert + BiLSTM + Dense）：將有犯罪事實標為1，與犯罪無關標為0的資料訓練。初步篩選包含犯罪之新聞。
2.	AML犯罪模型（Bert(微調) + Dense）：將犯罪且與AML有關標為1，犯罪且與AML無關標為0的資料訓練。篩選有AML相關犯罪之新聞。
3.	NER模型（Bert + BiLSTM + CRF）：用CKIP初步辨識並篩選出人名（包含三字、兩字簡稱及單名），以此訓練NER模型。
4.	人名AML模型（Bert(微調) + Dense）：取官方原始331筆包含AML人名新聞中所有人名的前後句訓練，將含有AML人名的前後句標為1，含有非AML人名的標為0。篩選最終AML人名。
## 特徵
1.	資料前處理：<br>
(1) 	將原始新聞刪除< >、【】、（）、〔〕中的字<br>
(2) 	排除記者…報導及「」中長度四以下的字，防止假名納入<br>
2.	犯罪模型、AML犯罪模型：<br>
(1) 	若長度大於512則取首尾256字預測<br>
3.	NER模型：<br>
(1) 	用維基百科內百家姓協助判斷並篩選正確人名<br>
4.	人名AML模型：<br>
(1) 	取包含該人名的句子，並加上前後各一句（以。，？；切）<br>
(2) 	若前後句遇到句點則取到該句為止，或從句點開始取<br>
(3) 	若前後句包含其他人名則捨棄該句<br>
(4) 	若中間句包含其他人名則置換成空字串<br>
5.	規則：<br>
(1) 	若中間句及後句含有：無罪定讞、確定無罪、無罪確定、罪嫌不足、罪證不足、不起訴則從預測結果排除<br>
(2) 	若中間句及後句人名前含有：檢察官、員警等職稱，人名後含有：說、調查、辦理、偵訊、訊問、諭令等字眼則從預測結果排除（此為CKIP斷詞後取其動詞統計後之結果）<br>


## 流程
![流程圖](https://github.com/jasonliu1990/esun_summer_game_2020/blob/master/docs/%E6%B5%81%E7%A8%8B.png)

## api 
* 環境: GCP ubuntu 18.04
* flask + gunicorn
* 主要package
  * tensorflow-gpu==1.15.3
  * Keras==2.3.1
  * keras-bert==0.84.0
* sudo gunicorn -w 1 --thread=4 -b 0.0.0.0:8080 --timeout=600 api:app --daemon
