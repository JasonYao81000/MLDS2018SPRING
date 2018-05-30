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
## Run bash to generate images.
```
bash run_gan.sh
```
## Test on baseline model.
```
cd gan-baseline
python3.6 baseline.py --input ../samples/gan_original.png
```
## Compare our model (WGAN_GP) with WGAN

|Train WGAN 50 epochs|
|:------------------:|
|![WGAN.gif](https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw3/hw3_1/results/WGAN_Anime_64_62/WGAN.gif)|

|Train WGAN_GP 50 epochs|
|:------------------:|
|![WGAN_GP.gif](https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw3/hw3_1/results/WGAN_GP_Anime_64_62/WGAN_GP.gif)|

## Training tips for improvement
**Tip 1** Normalize the images between -1 and 1.
**Tip 3** Use a spherical Z
**Tip 4** BatchNorm
**Tip 5** Avoid Sparse Gradients: ReLU, MaxPool
**Tip 14** Train discriminator more (sometimes)

# 3-2. Text-to-Image Generation
## Run bash to generate images.
```
bash run_cgan.sh
```
## Test on baseline model.
```
cd gan-baseline
python3.6 baseline.py --input ../samples/cgan_original.png
```

# 3-3. Style Transfer
```
bash extra_run.sh
```
