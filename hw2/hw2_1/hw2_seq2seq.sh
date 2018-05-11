#!/bin/bash
wget -O ./models/model7204-203.data-00000-of-00001 https://www.dropbox.com/s/9g01n49gtzkil1h/model7204-203.data-00000-of-00001?dl=1 
python3 infer.py $1 $2
