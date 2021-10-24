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
import pickle
from model import LSTM_Corona 
#↑あるクラスのインスタンスをpickle 化したとき、pickle を読み込んで利用するときにはクラスを別途import する必要がある。
#https://gist.github.com/masaki925/234c4d88228f987bee47bd47e2749fcb
from sklearn.preprocessing import MinMaxScaler

# #モデルの存在確認
MODEL_DIR_PATH = './model/bert' #モデル関連ディレクトリ
MODEL_FILE_PATH =  './model/bert/pytorch_model.bin' #モデル本体
CATEGORY_PATH = "./model/text"  # フォルダの場所を指定

covid19_data = './time_series_covid19_confirmed_global.csv'#COVID19データ
df = pd.read_csv(covid19_data)
df = df.iloc[:,37:]
daily_global = df.sum(axis=0)
daily_global.index = pd.to_datetime(daily_global.index)
y=daily_global.values.astype(float)

window_size = 30

def main():
    #タイトルの表示
    st.title('時系列処理')

    selected_item = st.selectbox('時系列処理を選択して下さい。',
                                 ['Covid19予測', '文章分類'])

    if selected_item == 'Covid19予測':
        st.write('・Covid19予測を行います。')

        out = analyze_lstm(10)

        #講座では2020年データ使用しており3日間しか予測しないので、グラフ化した時に見にくいので、１ヶ月分を予測するようにする。
        x = np.arange('2021-10-25','2021-11-04', dtype='datetime64[D]').astype('datetime64[D]')

        fig, ax = plt.subplots(figsize=(12,5))
        plt.title('The number of person affected by Corona virus globally')
        plt.grid(True)
        plt.plot(daily_global)#オリジナルデータ
        plt.plot(x, out[window_size:])#予測値
        st.pyplot(fig)

    else:
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
            dir_files = os.listdir(path=CATEGORY_PATH)

            for f in os.listdir(CATEGORY_PATH):
                # st.write(os.path.join(CATEGORY_PATH, f))
                if os.path.isdir(os.path.join(CATEGORY_PATH, f)):
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

#LSTM解析
def analyze_lstm(future=10):
    #モデルの読み込み
    with open("lstm.pickle", mode="rb") as f:
        model = pickle.load(f)
    
    #入力のデータを正規化（-1〜0に収まるように変換）
    scaler = MinMaxScaler(feature_range=(-1,1))

    y_normalized = scaler.fit_transform(y.reshape(-1,1))
    y_normalized = torch.FloatTensor(y_normalized).view(-1)
    preds = y_normalized[-window_size:].tolist()

    model.eval()#評価モード

    for i in range(future):
        sequence = torch.FloatTensor(preds[-window_size:])
    
        with torch.no_grad():#勾配の計算の無効化
            model.hidden =(torch.zeros(1,1,model.h_size),torch.zeros(1,1,model.h_size))#隠れ層NO無効化
            preds.append(model(sequence).item())#その都度計算された予測値を格納
        
    #予測値が正規化されてるので元のスケールに戻す。
    out = scaler.inverse_transform(np.array(preds).reshape(-1,1))

    return out

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