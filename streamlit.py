import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt 
import torch #BERT動かすにはtorchライブラリが必要。
from transformers import BertForSequenceClassification, BertJapaneseTokenizer
from torch import nn #ソフトマックス関数使用。
import pickle
from sklearn.preprocessing import MinMaxScaler
import japanize_matplotlib #matplotlibのラベルの文字化け解消のためインストール
import datetime
import matplotlib.ticker as mtick #グラフ描画時にy軸に%表示する。
from model import LSTM_Corona


# #モデルの存在確認
MODEL_DIR_PATH = './model/bert' #モデル関連ディレクトリ
MODEL_FILE_PATH =  './model/bert/pytorch_model.bin' #モデル本体
CATEGORY_PATH = "./model/text"  # フォルダの場所を指定

#基準年月日
base_y = 2021
base_m = 10
base_d = 24

window_size = 30

#COVID19データ取り込み
covid19_data = './time_series_covid19_confirmed_global.csv'
df = pd.read_csv(covid19_data)

#データの中で0で変化がないところを削る。
df = df.iloc[:,37:]

#日ベースごとに全世界を足して、各日ベースの世界全体の感染者数求める
daily_global = df.sum(axis=0)

#日付の値をpandasのdatetimeに変換
daily_global.index = pd.to_datetime(daily_global.index)

y = daily_global.values.astype(float)

def main():
    #タイトルの表示
    st.title('時系列処理')

    selected_item = st.selectbox('・時系列処理を選択して下さい。',
                                 ['', 'Covid19予測（LSTM）', '文章分類（BERT）'])
    
    if selected_item == 'Covid19予測（LSTM）':
        selected_item = st.selectbox('・何日後まで予測するか選択して下さい。',
                                 ['', '10日後', '20日後', '30日後', '60日後'])
        #予測ボタン
        start = st.button('予測開始')

        if start and not selected_item:
            st.write('<span style="color:red;">予測する期間を選択して下さい。</span>', unsafe_allow_html=True)

        if start and selected_item:
            date_dict = {'10日後':10, '20日後':20, '30日後':30, '60日後':60}
            sel_datte = date_dict[selected_item]

            d = datetime.date(base_y, base_m, base_d)
            target_d = d + datetime.timedelta(days=sel_datte)
            base_date = str(base_y) + '-' + str(base_m) + '-' + str(base_d) #基準日

            out = analyze_lstm(sel_datte)

            #講座では2020年データ使用しており3日間しか予測しないので、グラフ化した時に見にくいので、１ヶ月分を予測するようにする。
            x = np.arange(base_date,target_d, dtype='datetime64[D]').astype('datetime64[D]')

            fig, ax = plt.subplots(figsize=(12,5))
            plt.title('コロナウィルスの全世界感染者数')
            plt.ylabel("感染者数")
            plt.xlabel("日付")
            plt.grid(linestyle='dotted', linewidth=1)
            plt.gca().ticklabel_format(style='plain', axis='y')#y軸を省略せずにメモリ表示
            plt.plot(daily_global, label='確定値')#オリジナルデータ
            plt.plot(x, out[window_size:],label='予測結果')#予測値
            plt.legend(loc='best')
            st.pyplot(fig)

    elif selected_item == '文章分類（BERT）':
        text = st.text_area('・カテゴリー分類する記事を入力して下さい。')
        
        st.session_state.start = False

        #解析ボタン
        start = st.button('解析開始')
        
        if start:
            st.session_state.start = True

        #if start and not text:
        if st.session_state.start and not text:
            st.write('<span style="color:red;">解析する記事を入力して下さい。</span>', unsafe_allow_html=True)
       
        #if os.path.isdir(MODEL_DIR_PATH) and os.path.isfile(MODEL_FILE_PATH) and text and start:
        #if start and text:
        if st.session_state.start and text:
            
            #カテゴリー辞書
            category_dict = {'dokujo-tsushin':'独女通信',
                            'livedoor-homme':'livedoor HOMME',
                            'kaden-channel':'家電チャンネル',
                            'smax':'エスマックス',
                            'topic-news':'トピックニュース',
                            'peachy':'Peachy',
                            'movie-enter':'MOVIE ENTER',
                            'it-life-hack':'ITライフハック',
                            'sports-watch':'Sports Watch'}
            
            #カテゴリー辞書のキーリスト
            category_key_list = list(category_dict.keys())
            #カテゴリー辞書の値リスト
            category_values_list = list(category_dict.values())

            #改行\n、タブ\t、復帰\r、全角スペース\u3000を除去
            text = [sentence.strip() for sentence in text]
            text = list(filter(lambda line: line != '', text))
            text = ''.join(text)
            text = text.translate(str.maketrans({"\n":"", "\t":"", "\r":"", "\u3000":""})) 
            #解析実行
            out = analyze_bert(text)
            #出力結果が確率ではないためソフトマックスに通して確率にする。
            F = nn.Softmax(dim=1)
            out_F = F(out[0])

            #Tensor型からnumpyに変換→detach()関数でデータ部分を切り離し、numpy()でnumpyに変換する。
            predict = out_F.detach().numpy() 

            #解析結果の描画
            fig, ax = plt.subplots()
            x = np.arange(len(predict[0])) 
            y = np.round(predict[0]*100).astype(int)#結果を%表示するので四捨五入しint型に変換。

            plt.title("解析結果")
            plt.xlabel("カテゴリー", fontsize=13)
            plt.ylabel("確率", fontsize=13)
            plt.grid(linestyle='dotted', linewidth=1)
            plt.bar(x, y,  label='カテゴリー', align='center', alpha=0.7)
            plt.xticks(x, category_values_list, fontsize=8, rotation=45)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())#y軸を%表示
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
     #loaded_model = BertForSequenceClassification.from_pretrained(MODEL_DIR_PATH)
     #loaded_tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_DIR_PATH)

     loaded_model = load_bert_model()
     loaded_tokenizer = load_bert_tokenizer()
     max_length = 512
     words = loaded_tokenizer.tokenize(text)
     word_ids = loaded_tokenizer.convert_tokens_to_ids(words)  # 単語をインデックスに変換
     word_tensor = torch.tensor([word_ids[:max_length]])  # テンソルに変換
     out = loaded_model(word_tensor)
    
     return out

@st.cache(allow_output_mutation=True)
def load_bert_model():
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR_PATH)
    return model

@st.cache(allow_output_mutation=True)
def load_bert_tokenizer():
    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_DIR_PATH)
    return tokenizer

if __name__ == "__main__":
    main()
