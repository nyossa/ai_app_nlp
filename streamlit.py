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

# #モデルの存在確認
MODEL_DIR_PATH = './model' #モデル関連ディレクトリ
MODEL_FILE_PATH =  './model/pytorch_model.bin' #モデル本体

def main():
    #タイトルの表示
    st.title('自然言語処理')

    st.write('・入力された記事を解析して、「MOVIE ENTER」「ITライフハック」「家電チャンネル」「トピックニュース」「livedoor HOMME」「Peachy」「Sports Watch」「独女通信」「エスマックス」に分類します。')    
    st.write('・解析モデルを選んで下さい。')
    sel_bert = st.checkbox('BERT解析')
    sel_lstm = st.checkbox('LSTM解析')
    sel_random_forest = st.checkbox('Randamforest解析')
    text = st.text_area('・解析する記事を入力して下さい。')

    #解析ボタン
    start = st.button('解析')

    #選択された解析モデルの数
    num_sel = int(sel_bert) + int(sel_lstm) + int(sel_random_forest)

    #チェック
    if start and num_sel == 0:
        st.write('<span style="color:red;">解析モデルを選択して下さい。</span>', unsafe_allow_html=True)
    if start and not text:
        st.write('<span style="color:red;">解析する記事を入力して下さい。</span>', unsafe_allow_html=True)

    if os.path.isdir(MODEL_DIR_PATH) and os.path.isfile(MODEL_FILE_PATH) and text and num_sel != 0 and start:
        #モデルが配置されており、かつ入力がある場合

        # #モデル読み込み
        # loaded_model = BertForSequenceClassification.from_pretrained(MODEL_DIR_PATH)
        # # loaded_tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        # loaded_tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_DIR_PATH)

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
        text = [sentence.strip() for sentence in text]
        text = list(filter(lambda line: line != '', text))
        text = ''.join(text)
        text = text.translate(str.maketrans(
            {"\n":"", "\t":"", "\r":"", "\u3000":""})) 

        # max_length = 512
        # words = loaded_tokenizer.tokenize(text)
        # word_ids = loaded_tokenizer.convert_tokens_to_ids(words)  # 単語をインデックスに変換
        # word_tensor = torch.tensor([word_ids[:max_length]])  # テンソルに変換
        # out = loaded_model(word_tensor)
        out = analyze_bert(text)

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

        if sel_bert == True :
            plt.bar(x, predict[0], color='r', width=width, label='BERT', align='center')
        if sel_lstm == True :
            plt.bar(x+width, predict[0], color='b', width=width, label='LSTM', align='center')
        if sel_random_forest == True :
            plt.bar(x+width+width, predict[0], color='y', width=width, label='RandomForest', align='center')

        #x軸ラベル位置を調整
        plt.xticks(x + width, category_list, rotation=45)
        plt.legend(loc='best')
        st.pyplot(fig)

#BERT解析
def analyze_bert(text):

     #モデル読み込み
     loaded_model = BertForSequenceClassification.from_pretrained(MODEL_DIR_PATH)
     loaded_tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_DIR_PATH)

     max_length = 512
     words = loaded_tokenizer.tokenize(text)
     word_ids = loaded_tokenizer.convert_tokens_to_ids(words)  # 単語をインデックスに変換
     word_tensor = torch.tensor([word_ids[:max_length]])  # テンソルに変換
     out = loaded_model(word_tensor)
 
     return out

if __name__ == "__main__":
    main()