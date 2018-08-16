# グラフ描画のためのスクリプト

import pandas as pd  # csvファイルを扱うライブラリ
import matplotlib.pyplot as plt  # グラフ描画ライブラリ
import re  # 正規表現で使うライブラリ

ymax = 40  # グラフの縦軸の最大値
delimiter = 5  # グラフの横軸の区切る数字

train_name = 'train.log'  # 訓練データの結果のファイル名
val_name = 'val.log'  # 検証データの結果のファイル名

train = pd.read_table(train_name)  # 訓練データの精度のログファイルの読み込み
val = pd.read_table(val_name)  # 検証データの精度のログファイルの読み込み

train_acc = []  # 訓練データの精度を入れるリスト変数
val_acc = []  # 検証データの精度を入れるリスト変数

regex = r'[0-9]*\.[0-9]*'  # 文字列から少数を抽出する正規表現
length = len(train['acc'])  # epoch数

for i in range(length):  # epoch数繰り返す
    train_acc.append(float(re.search(regex, train['acc'][i]).group(0)) * 100)  # accの列から少数を抜き出しfloat型に変換して100を掛けてリストに追加
    val_acc.append(float(re.search(regex, val['acc'][i]).group(0)) * 100)

x = list(range(1, length + 1))  # グラフのx軸の設定

plt.plot(x, train_acc, label='train')  # 訓練データの精度の推移のグラフを、ラベル名「train」にして割り当てる
plt.plot(x, val_acc, linestyle='--', label='validation')  # 検証データの精度の推移のグラフを、ラベル名「validation」にして割り当てる
plt.legend()  # グラフのラベル名を図に表示する

plt.xlabel('epoch')  # x軸の名前を「epoch」にする
plt.ylabel('accuracy(%)')  # y軸の名前を「accuracy(%)」にする

plt.xlim(0, length + 1)  # x軸の範囲を指定
plt.ylim(0, ymax)  # y軸の範囲を指定

xscale = list(range(0, length + 1, delimiter))  # 0から20まで5刻みのリストを作成
xscale[0] = 1  # 最初は1から始まるようにする
plt.xticks(xscale)  # x軸の目盛りの設定

plt.show()  # 図の表示
