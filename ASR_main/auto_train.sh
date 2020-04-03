#!/bin/bash

for j in $(seq 0 2); do
 
  python3 -u main.py --model_iter $j
  dir_name="./VAEGAN_all";
  dir_name+=$j;
  dir_name+="/VAEGAN_all";
  dir_name+=$j;
 
  dir_name+="-200";
  python3 -u convert_stored.py --model_dir $dir_name
  
  cd converted_voices
  
  rm -r MCD_test
  
  mkdir MCD_test
  
  mv converted_* ./MCD_test
  
  cd ..
  python3 -u calculate_GV_vawgan.py --model_iter $j
  python3 -u calculate_MCD_vawgan.py --model_iter $j
  python3 -u calculate_MSD_vawgan.py --model_iter $j
done
