#!/bin/zsh

cd /home/svu/e0703350/Documents/projects/yoga_proj/yoga-project

CUDA_VISIBLE_DEVICES=0 python test_yoga_score_3item.py -r /hpctmp/e0703350/projects/yoga-saved/models/Yoga_Score_Net_filtered_data2_1dconv_arc_16-32-32-32-k15_noeasy/1031_093746/model_best.pth > logs/score_3item.log 2>&1 &