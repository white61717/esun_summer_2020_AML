# 玉山人工智慧公開挑戰賽2020夏季賽 - NLP應用挑戰賽
* Brainchild
## 預測說明
* 判斷該新聞內文是否含有AML相關焦點人物，並擷取出焦點人物名單
## 文件說明
* create_model.ipynb - 製作model
* api - 使用flask 將model佈署至GCP
## 摘要
用Bert分別訓練四階段模型
1.	犯罪模型（Bert + BiLSTM + Dense）：將有犯罪事實標為1，與犯罪無關標為0的資料訓練。初步篩選包含犯罪之新聞。
2.	AML犯罪模型（Bert(微調) + Dense）：將犯罪且與AML有關標為1，犯罪且與AML無關標為0的資料訓練。篩選有AML相關犯罪之新聞。
3.	NER模型（Bert + BiLSTM + CRF）：用CKIP初步辨識並篩選出人名（包含三字、兩字簡稱及單名），以此訓練NER模型。
4.	人名AML模型（Bert(微調) + Dense）：取官方原始331筆包含AML人名新聞中所有人名的前後句訓練，將含有AML人名的前後句標為1，含有非AML人名的標為0。篩選最終AML人名。
## 流程

