# RNNMutaion
Mutatuion analysis for RNN (LSTM and GRU). 2 types of mutation strategies are surported, static and dynamic. Static: mutate model weights and generate mutated model. Dynamic: mutate state at running time.  

#### How to use. File: src/runner.py
#### Parameter
* operator_type: static or dynamic
* model_path: original model path 
* save_path: model save path (static)
* operator: operator select
* single_data_path: dynamic time step mutation data (dynamic)
* layer_type: lstm or gru
* layer_name: lstm layer name
* rnn_cell_index: mutant rnn layer index out all rnn layers (dynamic)
* ratio: mutant ratio, default 0.01 
* gate_type: gate type, default 0
* precision_num: precision remain number, default 0
* standard_deviation: for gaussian fuzzing, default 0.0 
* time_stop_step: stop at which time step, default 0 (dynamic)
* time_start_step: re-start at which time step (for operator state reset), default 0 (dynamic)
* csv_path: dynamic testing results save path (static)
* num: index of mutants (static)
* acc_threshold: accuracy threshold

#### static testing example:
```
python runner.py --operator_type static --model_path ../models/imdb_gru.h5 --save_path ../generated_model --operator 8 --layer_type gru --layer_name gru_1 --ratio 0.30 --gate_type 4 --standard_deviation 0.1 --num 1 --acc_threshold 0.7725
```
#### dynamic testing example:
```
python runner.py --operator_type dynamic --model_path ../models/imdb_lstm.h5 --layer_type lstm --layer_name lstm_1 --rnn_cell_index 1 --operator 1 --single_data_path ../data/select_data.npz --standard_deviation 1.0 --precision_num 1 --time_stop_step 40 --csv_path "../result/test.csv"
```

#### mutation operator:
1. state status clear
2. state reset
3. state gaussian fuzzing
4. state precision reduction
5. dynamic gate clear
6. dynamic gate gaussian fuzzing
7. dynamic gate precision reduction
8. static gate gaussian fuzzing
9. static gate precision reduction
10. weight gaussian fuzzing
11. weight quantization
12. weight precision reduction

#### LSTM gate type:
0. input
1. forget
2. cell candidate
3. output
4. all gates
#### GRU gate type:
0. update
1. reset
2. cell candidate
3. all gates
  
