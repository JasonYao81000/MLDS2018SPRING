# MLDS2018SPRING/hw2/hw2_1
# Requirements
```
tensorflow-gpu==1.6.0
numpy==1.14.2
pandas==0.22.0
```
# Data Preprocessing
```
python3 data_preprocessing.py ./MLDS_hw2_1_data/training_data/feat/ ./MLDS_hw2_1_data/training_label.json
```
# Train Seq2Seq Model
```
rm ./models/*
python3 train.py ./MLDS_hw2_1_data/testing_data/feat/ ./MLDS_hw2_1_data/testing_label.json ./output_testset.txt
```
# Inference on Testing Set
```
python3 infer.py ./MLDS_hw2_1_data/testing_data ./testset_output.txt
```
If you just wanna infer without training, please run hw2_seq2seq.sh or download trained model by yourself.
```
bash ./hw2_seq2seq.sh ./MLDS_hw2_1_data/testing_data testset_output.txt
```
or
```
wget -O ./models/model7204-203.data-00000-of-00001 https://www.dropbox.com/s/9g01n49gtzkil1h/model7204-203.data-00000-of-00001?dl=1 
python3 infer.py ./MLDS_hw2_1_data/testing_data ./testset_output.txt
```
# Evaluate BLEU@1
```
python3 bleu_eval.py testset_output.txt
```
# Performance 
| Method                       | BLEU@1   |
| ---------------------------- |:--------:|
| Without Attention            | 0.7010   |
| Luong Attention              | 0.7054   |
| Luong Attention with scale   | 0.6977   |
| Bahdanau Attention           | 0.6965   |
| Bahdanau Attention with norm | 0.7204   |
