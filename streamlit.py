import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image #PIL（ピル）は画像表示用
##下記今回追加
import os
import matplotlib.pyplot as plt 
import torch #BERT動かすにはtorchライブラリが必要。
from transformers import BertForSequenceClassification, BertJapaneseTokenizer

#タイトルの表示
st.title('自然言語処理')

#モデルの存在確認
modelDirPath = './model' #モデル関連ディレクトリ
modelFilePath =  './model/pytorch_model.bin' #モデル本体

st.write('入力された記事を解析して、「MOVIE ENTER」「ITライフハック」「家電チャンネル」「トピックニュース」「livedoor HOMME」「Peachy」「Sports Watch」「独女通信」「エスマックス」に分類します。')    
article = st.text_area('記事を入力し、Ctrl+Enterで解析結果を表示します。')

if os.path.isdir(modelDirPath) and os.path.isfile(modelFilePath) and article:
    #モデルが配置されており、かつ入力がある場合

    #モデル読み込み
    loaded_model = BertForSequenceClassification.from_pretrained(modelDirPath)
    # loaded_tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    loaded_tokenizer = BertJapaneseTokenizer.from_pretrained(modelDirPath)

    category_list = []
    category_path = "./text"  # フォルダの場所を指定
    dir_files = os.listdir(path=category_path)

    for f in os.listdir(category_path):
        # st.write(os.path.join(category_path, f))
        if os.path.isdir(os.path.join(category_path, f)):
            category_list.append(f)
    
    #dirs = [f for f in dir_files if os.path.isdir(os.path.join(sample_path, f))]  # ディレクトリ一覧
    #st.write(category_list)

    # category_list = ['movie-enter', #MOVIE ENTER
    #                 'it-life-hack', #ITライフハック
    #                 'kaden-channel', #家電チャンネル
    #                 'topic-news',    #トピックニュース
    #                 'livedoor-homme', #livedoor HOMME
    #                 'peachy', # Peachy
    #                 'sports-watch', #Sports Watch
    #                 'dokujo-tsushin', #独女通信
    #                 'smax'] #エスマックス
 
    #改行\n、タブ\t、復帰\r、全角スペース\u3000を除去
    article = [sentence.strip() for sentence in article]
    article = list(filter(lambda line: line != '', article))
    article = ''.join(article)
    article = article.translate(str.maketrans(
        {"\n":"", "\t":"", "\r":"", "\u3000":""})) 

    max_length = 512
    words = loaded_tokenizer.tokenize(article)
    word_ids = loaded_tokenizer.convert_tokens_to_ids(words)  # 単語をインデックスに変換
    word_tensor = torch.tensor([word_ids[:max_length]])  # テンソルに変換
    pred = loaded_model(word_tensor)
    pred = pred[0].detach().numpy() #numpyに変換→detach()関数でデータ部分を切り離し、numpy()でnumpyに変換する。
    # st.write(pred[0])
    # st.write(pred[0].detach().numpy())
    st.write('解析結果：')
    st.write('BERT')
    st.table(pred )
    pred = np.round(pred, decimals=2) * 100
    # pred3 = np.round(pred, decimals=2) #四捨五入

    #描画
    fig, ax = plt.subplots()
    x = np.arange(len(pred[0])) 
    plt.title("Analysis result")
    plt.xlabel("category")
    plt.ylabel("probability")
    width = 0.3
    plt.bar(x, pred[0], color='r', width=width, label='BERT', align='center')
    plt.bar(x+width, pred[0], color='b', width=width, label='LSTM', align='center')
    plt.bar(x+width+width, pred[0], color='y', width=width, label='RandomForest', align='center')
    plt.xticks(x + width/3, category_list, rotation=45)
    plt.legend(loc='best')
    st.pyplot(fig)
