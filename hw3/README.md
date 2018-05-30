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
## Compare our model (WGAN_GP) with WGAN (50 epochs)

|      WGAN_GP       |        WGAN        |
|:------------------:|:------------------:|
|<img src="https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw3/hw3_1/results/WGAN_GP_Anime_64_62/WGAN_GP.gif" width="100%">|<img src="https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw3/hw3_1/results/WGAN_Anime_64_62/WGAN.gif" width="100%">|

## Training tips for improvement
### 1. Tip 1: Normalize the inputs 
- Normalize the images between -1 and 1 
- Tanh as the last layer of the generator output 
### 2. Tip 3: Use a spherical Z 
- Don't sample from a Uniform distribution 
- Sample from a gaussian distribution 
- When doing interpolations, do the interpolation via a great circle, rather than a straight line from point A to point B 
- Tom White's Sampling Generative Networks ref code https://github.com/dribnet/plat has more details 
### 3. Tip 4: BatchNorm 
- Construct different mini-batches for real and fake, i.e. each mini-batch needs to contain only all real images or all generated images. 
- When batchnorm is not an option use instance normalization (for each sample, subtract mean and divide by standard deviation). 
### 4. Tip 5: Avoid Sparse Gradients: ReLU, MaxPool 
- The stability of the GAN game suffers if you have sparse gradients
- LeakyReLU = good (in both G and D)
- For Downsampling, use: Average Pooling, Conv2d + stride
- For Upsampling, use: PixelShuffle, ConvTranspose2d + stride
  - PixelShuffle: https://arxiv.org/abs/1609.05158
### 5. Tip 14: Train discriminator more (sometimes) 
- Especially when you have noise
- Hard to find a schedule of number of D iterations vs G iterations

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
