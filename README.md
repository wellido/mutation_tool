# mutation_tool

#### this mutation tool supports for both CNN and RNN models
##### CNN mutation tool -> folder: cnn_mutation
##### RNN mutation tool -> folder: rnn_mutation


# project structure
```
├── cnn_mutation                  # cnn mutation folder
│   ├── src                       # code folder
│   │   ├── cnn_operator.py       # define cnn mutation operators
│   │   ├── data_sort.py          # sort data by mutants killed number
│   │   ├── generator.py          # mutants generator
│   │   ├── run.sh                # shell for test
│   │   ├── utils.py              # some useful functions
│   │   ├── __init__.py           # 
│   ├── __init__.py               #
│   ├── README.md                 # 
├── data                          # save some test data
├── models                        # save some test models
├── result                        # save some test results
├── rnn_mutation                  # rnn mutation folder
│   ├── src                       # code folder
│   │   ├── keras                 # keras library
│   │   │   ├── ...               #
│   │   ├── __init__.py           #
│   │   ├── gru_operator.py       # define gru mutation operators
│   │   ├── lstm_operator.py      # define lstm mutation operators
│   │   ├── run.sh                # shell for test
│   │   ├── runner.py             # mutants generator
│   │   ├── sort_segment.py       # sort segment by distance
│   │   ├── state_save.py         # save state
│   │   ├── utils.py              # some useful functions
│   ├── README.md                 #
│   ├── __init__.py               #
├── LICENSE                       # MIT license
├── README.md                     #
├── test.py                       # for my test
├── utils.py                      # some useful functions
```
