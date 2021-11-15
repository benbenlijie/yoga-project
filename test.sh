#!/bin/bash

cfg_pattern="config"

function loop_dir(){
    count=2
    for file in `ls $1`
    do
      if [ -f $1"/"$file ] && [[ $file == $cfg_pattern* ]]
      then

        count=$count-1
        echo $file $count
        sleep 3 &

      fi

      if [[ $count -eq 0 ]]
      then
        wait
        count=2
      fi
    done
}

loop_dir $1
