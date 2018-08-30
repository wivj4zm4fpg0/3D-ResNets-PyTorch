# グラフ描画のためのスクリプト

import argparse

import matplotlib.pyplot as plt  # グラフ描画ライブラリ
import pandas as pd  # csvファイルを扱うライブラリ

parser = argparse.ArgumentParser()
parser.add_argument(
    'input_train', default=None, type=str, help='input train file'
)
parser.add_argument(
    'input_val', default=None, type=str, help='input validation file'
)
parser.add_argument(
    '--name1', default='train', type=str, help='first graph name'
)
parser.add_argument(
    '--name2', default='validation', type=str, help='second graph name'
)
args=parser.parse_args()


y_axis_max = 0  # グラフの縦軸の最大値
delimiter = 5  # グラフの横軸の区切る数字

train_name = args.input_train  # 訓練データの結果のファイル名
val_name = args.input_val  # 検証データの結果のファイル名

train = pd.read_table(train_name)  # 訓練データの精度のログファイルの読み込み
val = pd.read_table(val_name)  # 検証データの精度のログファイルの読み込み

train_acc_list = []  # 訓練データの精度を入れるリスト変数
val_acc_list = []  # 検証データの精度を入れるリスト変数

epoch_length = len(train['acc'])  # epoch数

for i in range(epoch_length):  # epoch数繰り返す
    train_acc_list.append(train['acc'][i] * 100)  # accの列から少数を抜き出しfloat型に変換して100を掛けてリストに追加
    val_acc_list.append(val['acc'][i] * 100)

x = list(range(1, epoch_length + 1))  # グラフのx軸の設定

max_value = max(max(train_acc_list), max(val_acc_list)) # ２つのリストの最大値を取得
y_axis_max = ((int('{0:02d}'.format(int(max_value))[0]) + 1) * 10) # ２桁の数値の２の位に１を足して１０を掛けることによりy軸の最大値を決定する
print(max_value)

plt.plot(x, train_acc_list, label=args.name1)  # 訓練データの精度の推移のグラフを、ラベル名「train」にして割り当てる
plt.plot(x, val_acc_list, linestyle='--', label=args.name2)  # 検証データの精度の推移のグラフを、ラベル名「validation」にして割り当てる
plt.legend()  # グラフのラベル名を図に表示する

plt.xlabel('epoch')  # x軸の名前を「epoch」にする
plt.ylabel('accuracy(%)')  # y軸の名前を「accuracy(%)」にする

plt.xlim(0, epoch_length + 1)  # x軸の範囲を指定
plt.ylim(0, y_axis_max)  # y軸の範囲を指定

xscale = list(range(0, epoch_length + 1, delimiter))  # 0から20まで5刻みのリストを作成
xscale[0] = 1  # 最初は1から始まるようにする
plt.xticks(xscale)  # x軸の目盛りの設定

plt.show()  # 図の表示
