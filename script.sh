#!/usr/bin/env bash

# 10 Times
for i in $( seq 1 10 )
do
    echo "$i"
    CUDA_VISIBLE_DEVICES=1 nohup python attack_imagenet.py --iter_num=$i --model=resnet50 >>resnet50.out 2>&1 &
    CUDA_VISIBLE_DEVICES=3 python attack_imagenet.py --iter_num=$i --model=vgg16 >>vgg16.out 2>&1
    CUDA_VISIBLE_DEVICES=3 python attack_imagenet.py --iter_num=$i --model=inceptionv3
done