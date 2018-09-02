#!/bin/bash -eu

python main.py --video_path /home/satoru/dataset/UCF-101/resize/images \
--annotation_path /home/satoru/dataset/UCF-101/ucfTrainTestlist/ucf101_01.json \
--result_path /home/satoru/1tb/deep-learning-log \
--dataset ucf101 \
--n_classes 400 \
--ft_begin_index 0 \
--model resnet \
--model_depth 18 \
--resnet_shortcut A \
--batch_size 32 \
--n_threads 4 \
--checkpoint 1 \
--learning_rate 0.01 \
--flow_x_path /home/satoru/dataset/UCF-101/resize/flow/tvl1/flow_x \
--flow_y_path /home/satoru/dataset/UCF-101/resize/flow/tvl1/flow_y \
--n_epochs 30 \
--resume_path save_17.pth

python main.py --video_path /home/satoru/dataset/UCF-101/resize/images \
--annotation_path /home/satoru/dataset/UCF-101/ucfTrainTestlist/ucf101_01.json \
--result_path /home/satoru/1tb/deep-learning-log \
--dataset ucf101 \
--n_classes 400 \
--ft_begin_index 0 \
--model resnet \
--model_depth 18 \
--resnet_shortcut A \
--batch_size 32 \
--n_threads 4 \
--checkpoint 1 \
--learning_rate 0.001 \
--flow_x_path /home/satoru/dataset/UCF-101/resize/flow/tvl1/flow_x \
--flow_y_path /home/satoru/dataset/UCF-101/resize/flow/tvl1/flow_y \
--n_epochs 50 \
--resume_path save_30.pth