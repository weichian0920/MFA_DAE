# Blind Monaural Source Separation on Heart and Lung Sounds Based on Periodic-Coded Deep Autoencoder
### Introduction
This is the implementation of [Blind Monaural Source Separation on Heart and Lung Sounds Based on Periodic-Coded Deep Autoencoder](https://ieeexplore.ieee.org/document/9167389).
<p align="center">
  <img src="https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/6221020/9248684/9167389/tsao1-3016831-large.gif" width="400" height="300"/>
  <img src="https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/6221020/9248684/9167389/tsao4-3016831-large.gif" width="400" height="300"/>
</p>

### Features
* Model
  * There are two type autoencoders, including DAE_C(convolutional module) and DAE_F(Fully-connected module).
* MFA tool
  * The tool for source separation by modulating latent code of autoencoder.
### Requirements
* librosa             0.8.0. 
* torch               1.7.0
* torchvision         0.8.2
* sklearn             0.0
* numpy               1.19.5
* scipy               1.6.0
### Example Data
There is an example heart-lung sound 0_0.wav in ./src/dataset/ folder.
### How to run
Scripts to reproduce the training and evaluation procedures discussed in the paper are located on scripts/.
### Citation
If you find the code helpful in your research, please do consider cite us!
```bash
@article{tsai2020blind,
  title={Blind Monaural Source Separation on Heart and Lung Sounds Based on Periodic-Coded Deep Autoencoder},
  author={Tsai, Kun-Hsi and Wang, Wei-Chien and Cheng, Chui-Hsuan and Tsai, Chan-Yen and Wang, Jou-Kou and Lin, Tzu-Hao and Fang, Shih-Hau and Chen, Li-Chin and Tsao, Yu},
  journal={IEEE Journal of Biomedical and Health Informatics},
  volume={24},
  number={11},
  pages={3203--3214},
  year={2020},
  publisher={IEEE}
}
```
