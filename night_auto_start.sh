#! /bin/bash

while true 
do
	monitoring=` ps -ef|grep train.py |grep easy_margin |grep -v grep| wc -l`
	if [ $monitoring -eq 0 ] 
	then
		nohup python train.py -c config_base_weight_decay.json >> logs/weight_decay.log 2>&1 &
	fi
	sleep 3
done
