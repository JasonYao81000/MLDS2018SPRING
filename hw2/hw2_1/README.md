# MLDS2018SPRING/hw2/hw2_1
# Data Preprocessing
```
python3 data_preprocessing.py ./MLDS_hw2_1_data/training_data/feat/ ./MLDS_hw2_1_data/training_label.json
```
# Train Seq2Seq Model
```
python3 train.py ./MLDS_hw2_1_data/testing_data/feat/ ./MLDS_hw2_1_data/testing_label.json ./output_testset.txt
```
# Inference on Testing Set
```
python3 infer.py ./MLDS_hw2_1_data/testing_data ./testset_output.txt
```
If you just wanna infer without training, please run hw2_seq2seq.sh.
```
bash ./hw2_seq2seq.sh testing_data testset_output.txt
```
# Evaluate BLEU@1
```
python3 bleu_eval.py testset_output.txt
```
