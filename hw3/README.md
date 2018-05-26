# MLDS2018SPRING/hw3
# 3-0. Requirements
```
tensorflow-gpu==1.6.0
numpy==1.14.2
scipy==1.1.0
six==1.10.0
matplotlib==2.2.2
opencv-python==3.4.0.12
```
# 3-1. Image Generation
Train WGAN_GP with 10 epochs, and then infer.
```
bash run_gan.sh
```
Test on baseline model.
```
cd gan-baseline
python3.6 baseline.py --input ../samples/gan_original.png
```
# 3-2. Text-to-Image Generation
```
bash run_cgan.sh
```
Test on baseline model.
```
cd gan-baseline
python3.6 baseline.py --input ../samples/cgan_original.png
```
# 3-3. Style Transfer
```
bash extra_run.sh
```
