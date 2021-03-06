#!/bin/bash

data_dir="/home/ubuntu/drive2/kinetics400_30fps_frames"
output_dir="./model"
eval_dir="./model/eval_svm"
pretrained="pretrain/moco_v2_200ep_pretrain.pth.tar"
num_replica=4

mkdir -p ${output_dir}
mkdir -p ${eval_dir}

python3 -m torch.distributed.launch --master_port 9999 --nproc_per_node=${num_replica} \
    eval_svm_feature_extract.py \
    --data_dir=${data_dir} \
    --datasplit=train \
    --pretrained_model=${output_dir}/current.pth \
    --output_dir=${eval_dir}

python3 -m torch.distributed.launch --master_port 9999 --nproc_per_node=${num_replica} \
    eval_svm_feature_extract.py \
    --data_dir=${data_dir} \
    --datasplit=val \
    --pretrained_model=${output_dir}/current.pth \
    --output_dir=${eval_dir}


python3 eval_svm_feature_perf.py \
    --trainsplit=train \
    --valsplit=val \
    --output-dir=${eval_dir} \
    --num_replica=${num_replica} \
    --primal