#!/usr/bin/env bash
python generator.py --model_path ../../models/mnist_lenet5.h5 --data_type mnist --threshold 0.9 --operator 0 --ratio 0.05 --save_path ../../../lenet5-mutants2 --num 200
#python generator.py --model_path ../../models/mnist_lenet5.h5 --operator 6 --ratio 0.03 --save_path ../../../lenet5-mutants --num 100
#python generator.py --model_path ../../models/mnist_lenet5.h5 --operator 6 --ratio 0.05 --save_path ../../../lenet5-mutants --num 100