#!/bin/bash

python main.py --video_path \
/home/satoru/dataset/TwentyBN/20bn-something-something-v2/image_resize \
--result_path \
/home/satoru/1tb/results \
--dataset \
ssv2flow \
--n_classes \
400 \
--n_finetune_classes \
174 \
--pretrain_path \
pretrain_models/resnet-18-kinetics.pth \
--ft_begin_index \
0 \
--model \
resnet \
--model_depth \
18 \
--resnet_shortcut \
A \
--train_batch_size \
8 \
--n_threads \
12 \
--checkpoint \
1 \
--learning_rate \
0.001 \
--something_label_path \
/home/satoru/dataset/TwentyBN/20bn-something-something-v2/label/something-something-v2-labels.json \
--something_train_path \
/home/satoru/dataset/TwentyBN/20bn-something-something-v2/label/shapedtrain.json \
--something_val_path \
/home/satoru/dataset/TwentyBN/20bn-something-something-v2/label/shapedval.json \
--something_test_path \
/home/satoru/dataset/TwentyBN/20bn-something-something-v2/label/shapedtest.json \
--val_batch_size \
8 \
--optical_flow \
--flow_x_path \
/home/satoru/dataset/TwentyBN/20bn-something-something-v2/flow_x_resize \
--flow_y_path \
/home/satoru/dataset/TwentyBN/20bn-something-something-v2/flow_y_resize
