# グラフ描画のためのスクリプト

import argparse

import matplotlib.pyplot as plt  # グラフ描画ライブラリ
import pandas as pd  # csvファイルを扱うライブラリ

parser = argparse.ArgumentParser()
parser.add_argument(
    'input1_train', default=None, type=str, help='input train file'
)
parser.add_argument(
    'input1_val', default=None, type=str, help='input validation file'
)
parser.add_argument(
    'input2_train', default=None, type=str, help='compared input train file'
)
parser.add_argument(
    'input2_val', default=None, type=str, help='input input validation file'
)
parser.add_argument(
    '--name1', default='train', type=str, help='first g raph name'
)
parser.add_argument(
    '--name2', default='validation', type=str, help='second graph name'
)
parser.add_argument(
    '--parameter_name', default='acc-top1', type=str, help='a parameter to be compared'
)
parser.add_argument(
    '--delimiter', default=5, type=int, help='units to break graphs'
)
args = parser.parse_args()

y_axis_max = 0  # グラフの縦軸の最大値
delimiter = args.delimiter  # グラフの横軸の区切る数字

train_name1 = args.input1_train
val_name1 = args.input1_val

train_name2 = args.input2_train
val_name2 = args.input2_val

train1 = pd.read_table(train_name1)
val1 = pd.read_table(val_name1)

train2 = pd.read_table(train_name2)
val2 = pd.read_table(val_name2)

param = args.parameter_name  # 比較するパラメータ

acc_sub_list1 = []
acc_sub_list2 = []

epoch_length = min(len(train1[param]), len(val1[param]), len(train2[param]), len(val2[param]))  # epoch数

for i in range(epoch_length):  # epoch数繰り返す
    acc_sub_list1.append(
        abs(train1[param][i] - val1[param][i]) * 100)  # accの列から少数を抜き出しfloat型に変換して100を掛けてリストに追加
    acc_sub_list2.append(abs(train2[param][i] - val2[param][i]) * 100)

x = list(range(1, epoch_length + 1))  # グラフのx軸の設定

max_value = max(max(acc_sub_list1), max(acc_sub_list2))  # ２つのリストの最大値を取得
y_axis_max = ((int('{0:02d}'.format(int(max_value))[0]) + 1) * 10)  # ２桁の数値の２の位に１を足して１０を掛けることによりy軸の最大値を決定する

plt.plot(x, acc_sub_list1, label=args.name1)
plt.plot(x, acc_sub_list2, linestyle='--', label=args.name2)
plt.legend()  # グラフのラベル名を図に表示する

plt.xlabel('epoch')  # x軸の名前を「epoch」にする
plt.ylabel('accuracy(%)')  # y軸の名前を「accuracy(%)」にする

plt.xlim(0, epoch_length + 1)  # x軸の範囲を指定
plt.ylim(0, y_axis_max)  # y軸の範囲を指定

xscale = list(range(0, epoch_length + 1, delimiter))  # 0から20まで5刻みのリストを作成
xscale[0] = 1  # 最初は1から始まるようにする
plt.xticks(xscale)  # x軸の目盛りの設定

plt.show()  # 図の表示
