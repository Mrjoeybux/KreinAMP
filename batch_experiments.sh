#!/usr/bin/env bash

#python main.py comparison --dataset ampscan
#python main.py comparison --dataset deepamp
#python main.py comparison --dataset SA25923_MEDIAN
#python main.py comparison --dataset SA29213_MEDIAN
#python main.py comparison --dataset PA27853_MEDIAN

#python main.py comparison --dataset ampscan --num_outer 1
#python main.py comparison --dataset deepamp --num_outer 1
python main.py comparison --dataset SA25923_MEDIAN --num_outer 1
python main.py comparison --dataset SA29213_MEDIAN --num_outer 1
python main.py comparison --dataset PA27853_MEDIAN --num_outer 1

#python main.py predict --dataset ampscan
#python main.py predict --dataset deepamp

#shutdown -h now