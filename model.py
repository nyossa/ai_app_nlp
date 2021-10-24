import torch.nn as nn

class LSTM_Corona(nn.Module):
    def __init__(self, in_size=1, h_size=30, out_size=1):
        super().__init__()
        self.h_size = h_size
        self.lstm = nn.LSTM(in_size,h_size)
        self.fc = nn.Linear(h_size,out_size)
        
        self.hidden = (torch.zeros(1,1,h_size),torch.zeros(1,1,h_size))
        
    def forward(self, sequence_data):
        #lstmを実行するときは3次元のサイズを指定する必要がある。
        #１つ目の引数＝1次元目＝データのサイズ（len(sequence_data)）＝今回は30個 = train_dataNの中は30
        #2つ目の引数＝2次元目＝バッチサイズ＝今回はバッチ化していないので1
        #3つ目の引数＝3次元目＝隠れ層のサイズ＝今回なら引数で指定した30。
        lstm_out, self.hidden = self.lstm(sequence_data.view(len(sequence_data),1,-1),self.hidden)
        pred=self.fc(lstm_out.view(len(sequence_data),-1))
        
        return pred[-1]#欲しいのは最後のデータ