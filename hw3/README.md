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
```
bash run_gan.sh
```
Test on baseline model.
```
cd gan-baseline
python3.6 baseline.py --input ../samples/gan_original.png
```
![WGAN.gif](https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw3/hw3_1/results/WGAN_Anime_64_62/WGAN.gif)
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
