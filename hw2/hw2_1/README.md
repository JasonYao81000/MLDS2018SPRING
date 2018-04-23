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
wget -O ./models/model7204-203.data-00000-of-00001 https://www.dropbox.com/s/9g01n49gtzkil1h/model7204-203.data-00000-of-00001?dl=1 
python3 infer.py ./MLDS_hw2_1_data/testing_data ./testset_output.txt
```
# Evaluate BLEU@1
```
python3 bleu_eval.py testset_output.txt
```
