import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image #PIL（ピル）は画像表示用
##下記今回追加
import os
import matplotlib.pyplot as plt 
import torch #BERT動かすにはtorchライブラリが必要。
from transformers import BertForSequenceClassification, BertJapaneseTokenizer
from torch import nn #ソフトマックス関数使用。

#タイトルの表示
st.title('自然言語処理')

#モデルの存在確認
modelDirPath = './model' #モデル関連ディレクトリ
modelFilePath =  './model/pytorch_model.bin' #モデル本体

st.write('・入力された記事を解析して、「MOVIE ENTER」「ITライフハック」「家電チャンネル」「トピックニュース」「livedoor HOMME」「Peachy」「Sports Watch」「独女通信」「エスマックス」に分類します。')    
st.write('・解析するモデルを選んで下さい。')
isBert = st.checkbox('BERT解析')
isLstm = st.checkbox('LSTM解析')
isRandamforest = st.checkbox('Randamforest解析')
article = st.text_area('・記事を入力し、Ctrl+Enterで解析結果を表示します。')

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
    out = loaded_model(word_tensor)

    #出力結果が確率ではないためソフトマックスに通して確率にする。
    F = nn.Softmax(dim=1)
    out_F = F(out[0])

    #Tensor型からnumpyに変換→detach()関数でデータ部分を切り離し、numpy()でnumpyに変換する。
    predict = out_F.detach().numpy() 

    #結果の描画
    st.write('解析結果：')
    fig, ax = plt.subplots()
    x = np.arange(len(predict[0])) 
    plt.title("Analysis result")
    plt.xlabel("category")
    plt.ylabel("probability")
    width = 0.3

    #チェックされたモデル数によって描画調整
    checked = int(isBert) + int(isLstm) + int(isRandamforest)

    if isBert == True :
        plt.bar(x, predict[0], color='r', width=width, label='BERT', align='center')
    if isLstm == True :
        plt.bar(x+width, predict[0], color='b', width=width, label='LSTM', align='center')
    if isRandamforest == True :
        plt.bar(x+width+width, predict[0], color='y', width=width, label='RandomForest', align='center')
    
    #x軸ラベル位置を調整
    plt.xticks(x + width, category_list, rotation=45)
    plt.legend(loc='best')
    st.pyplot(fig)
