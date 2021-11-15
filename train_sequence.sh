#!/bin/bash

cd /home/svu/e0703350/Documents/projects/yoga_proj/yoga-project/

basic_config_file="config_filtered_data_1dconvTEMP.json"
log_file="logs/filtered_1dconvTEMP.log"

for (( i=7; i<10; i++ ));
do
    conf=${basic_config_file//TEMP/$i}
    logf=${log_file//TEMP/$i}
    echo $conf
    echo $logf
    # python train.py -c ${conf}
    nohup python train.py -c ${conf} > ${logf} 2>&1 &
    wait
done

# CUDA_VISIBLE_DEVICES=0 nohup python train.py -c config_filtered_data_1dconv6.json > logs/filtered_1dconv6.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py -c config_filtered_data_1dconv7.json > logs/filtered_1dconv7.log 2>&1 &

# wait
