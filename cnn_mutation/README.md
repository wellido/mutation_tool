# CNN-Mutation

#### Mutants generation: src/generator.py

```
python generator.py [-model_path FILE] [-operator INT] [-ratio FLOAT]
                    [-save_path DIR] [-num INT] [-data_type STRING]
                    [-threshold FLOAT] [-standard_deviation FLOAT]

optional arguments:
-model_path:          original model path
-operator:            mutation mutator
-ratio:               mutation ratio
-save_path:           mutants save path
-num:                 mutants generation number
-data_type:           mnist or cifar-10
-threshold:           mutant accuracy > original mutant * threshold
-standard_deviation:  Gaussian fuzzing standard deviation
```

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

#### Sort and generate report
```
API  sort_data(ori_model_path, mutants_path, x, y, save_path)
      """
      :param ori_model_path: path of the original model
      :param mutants_path: path of the mutants folder
      :param x: test data
      :param y: label of x
      :param save_path: report save path
      """
Directly use 
      python sort_data.py [-model_path FILE] [-mutants_path FOLDER] [-model_path FILE]
will use MNIST data default.

```
