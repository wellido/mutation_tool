# CNN-Mutation

#### Mutants generation
##### File: src/generator.py

#### Parameter
* model_path: original model path
* operator: mutation mutator
* ratio: mutation ratio
* save_path: mutants save path
* num: mutants generation number
* data_type: mnist or cifar-10
* threshold: mutant accuracy > original mutant * threshold

#### Mutation operators
0. gaussian fuzzing
1. weights shuffle
2. neuron block
3. neuron activation inverse
4. neuron switch
5. layer remove
6. layer addition
7. layer duplication

#### example
```
python generator.py --model_path models/mnist_lenet5.h5 --operator 0 --ratio 0.01 --save_path ../mutants --num 2 --data_type mnist --threshold 0.9
```

#### Data sort
##### File: src/data_sort.py

#### Parameter
* ori_model_path: original model path
* mutants_path: path of mutants folder
* x: x_test
* y: y_test
* save_path: sorted indices and killed number
