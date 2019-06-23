#!/usr/bin/env bash
#python generator.py --model_path ../../models/mnist_lenet5.h5 --data_type mnist --threshold 0.9 --operator 1 --ratio 0.01 --save_path ../../../lenet5-WS0.01 --num 200
#python generator.py --model_path ../../models/mnist_lenet5.h5 --data_type mnist --threshold 0.9 --operator 1 --ratio 0.03 --save_path ../../../lenet5-WS0.03 --num 200
#python generator.py --model_path ../../models/mnist_lenet5.h5 --data_type mnist --threshold 0.9 --operator 1 --ratio 0.05 --save_path ../../../lenet5-WS0.05 --num 200
#python generator.py --model_path ../../models/mnist_lenet5.h5 --data_type mnist --threshold 0.9 --operator 2 --ratio 0.01 --save_path ../../../lenet5-NEB0.01 --num 200
#python generator.py --model_path ../../models/mnist_lenet5.h5 --data_type mnist --threshold 0.9 --operator 2 --ratio 0.03 --save_path ../../../lenet5-NEB0.03 --num 200
#python generator.py --model_path ../../models/mnist_lenet5.h5 --data_type mnist --threshold 0.9 --operator 2 --ratio 0.05 --save_path ../../../lenet5-NEB0.05 --num 200
#python generator.py --model_path ../../models/mnist_lenet5.h5 --data_type mnist --threshold 0.9 --operator 3 --ratio 0.01 --save_path ../../../lenet5-NAI0.01 --num 200
#python generator.py --model_path ../../models/mnist_lenet5.h5 --data_type mnist --threshold 0.9 --operator 3 --ratio 0.03 --save_path ../../../lenet5-NAI0.03 --num 200
#python generator.py --model_path ../../models/mnist_lenet5.h5 --data_type mnist --threshold 0.9 --operator 3 --ratio 0.05 --save_path ../../../lenet5-NAI0.05 --num 200
#python generator.py --model_path ../../models/mnist_lenet5.h5 --data_type mnist --threshold 0.9 --operator 4 --ratio 0.01 --save_path ../../../lenet5-NS0.01 --num 200
#python generator.py --model_path ../../models/mnist_lenet5.h5 --data_type mnist --threshold 0.9 --operator 4 --ratio 0.03 --save_path ../../../lenet5-NS0.03 --num 200
python generator.py --model_path ../../models/mnist_lenet5.h5 --data_type mnist --threshold 0.9 --operator 4 --ratio 0.05 --save_path ../../../lenet5-NS0.05 --num 200
#python generator.py --model_path ../../models/mnist_lenet5.h5 --operator 6 --ratio 0.03 --save_path ../../../lenet5-mutants --num 100
#python generator.py --model_path ../../models/mnist_lenet5.h5 --operator 6 --ratio 0.05 --save_path ../../../lenet5-mutants --num 100