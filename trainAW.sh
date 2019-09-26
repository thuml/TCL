#!/usr/bin/env bash

python main.py \
    --arch resnet50 --pretrained  \
    --classes 31 \
    --bottleneck 256 \
    --gpu 6 \
    --batch-size 32 \
    --print-freq 250 --train-iter 20000 --test-iter 500 \
    --name AW-0.003-20000-0.4noise \
    --lr 0.003 \
    --dataset office --traindata amazon --valdata webcam \
    --noiselevel 0.4 --noisetype noise \
    --traded 1.0 --tradet 0.1 --startiter 3000 --Lythred 0.3 --Ldthred 0.5 --lambdad 0.1

