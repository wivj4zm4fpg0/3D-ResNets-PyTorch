import torch
import torch.nn as nn

ADD_FRAMES = 2

conv3 = nn.Conv2d(3, 6, 3, 3)
conv5 = nn.Conv2d(3 + ADD_FRAMES, 6, 3, 3)
data3 = conv3.weight.data
data5 = conv5.weight.data

for i in range(len(data3)):
    for j in range(len(data3[i])):
        data5[i][j] = data3[i][j]
    avg = torch.sum(data3[i], 0) / 3
    
    for j in range(ADD_FRAMES):
        data5[i][-j - 1] = avg

print(conv3.weight)
print()
print(conv5.weight)

