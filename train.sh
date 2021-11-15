#!/bin/bash

cd /home/svu/e0703350/Documents/projects/yoga_proj/yoga-project/

basic_config_file="config_filtered_data_1dconvTEMP.json"
log_file="logs/filtered_1dconvTEMP.log"
cuda_device=0

for (( i=13; i<18; i++ ));
do
    conf=${basic_config_file//TEMP/$i}
    logf=${log_file//TEMP/$i}
    echo $conf
    echo $logf
    echo $cuda_device
    CUDA_VISIBLE_DEVICE=$cuda_device nohup python train.py -c ${conf} > ${logf} 2>&1 &
    cuda_device=$[$cuda_device+1]
    if [ $cuda_device -ge 2 ];
    then
        cuda_device=0
        wait
    fi
    
done
