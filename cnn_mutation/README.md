# CNN-Mutaion
#### How to use. File: src/generator.py

#### Parameter
* model_path: original model path
* operator: mutation mutator
* ratio: mutation ratio
* save_path: mutants save path
* num: mutants generation number

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
python generator.py --model_path models/mnist_lenet5.h5 --operator 0 --ratio 0.01 --save_path ../mutants --num 2
```
