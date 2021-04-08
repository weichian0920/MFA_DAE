# Blind Monaural Source Separation on Heart and Lung Sounds Based on Periodic-Coded Deep Autoencoder
### Introduction
This is the implementation of [Blind Monaural Source Separation on Heart and Lung Sounds Based on Periodic-Coded Deep Autoencoder](https://ieeexplore.ieee.org/document/9167389) by pytorch. The system flow is shown as:
<p align="center">
  <img src="https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/6221020/9248684/9167389/tsao1-3016831-large.gif" width="400" height="300"/>
  <img src="https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/6221020/9248684/9167389/tsao4-3016831-large.gif" width="400" height="300"/>
</p>


Fisrt we need to train an autoencoder, then use our MFA tool to analysis and modulate latent space code to accomplish blind monaural source separation.

### Features
* Audio processing
  *  We adopt librosa 0.8.0 audio processing tool to process the audio.
  *  The Fourier transform variables can be defined using dictionary FFT_dict.
    > For example:
  ```bash
              FFT_dict = {
                            'sr': 8000,
                            'frequency_bins': [0, 300],
                            'FFTSize': 2048,
                            'Hop_length': 128,
                            'Win_length': 2048,
                            'normalize': True,
                          }
  ```
  * hl_dataloader
    The dataloader construct by pytorch.
    > user need provide a list of .wav data for dataloader.
    
* Model
  * There are two type of autoencoders, including DAE_C(convolutional module) and DAE_F(Fully-connected module). The default architectures are defined in model_dict.
  *  Defining Models
   > Models can be succinctly defined using dictionary by combining its variables, layers and scopes. Each of these elements is defined below.
   > For example(DAE_F):
     ```bash
              model_dict = {
                            "frequency_bins":[0, 257], # The range of input of log power spectrum frequency bin. e.g. the dimension of input is (batch, 257-0)
                            "encoder":[128, ..., 16], # The length of list is the encoder layers, each item in list is neurons for each layers of encoder.
                            "decoder":[16, ..., 257], # The length of list is the decoder layers, each item in list is neurons for each layers of decoder.
                            "encoder_act": string, # activation function for encoder.
                            "decoder_act": string, # activation function for decoder.
                            }
     ```   
  
  * Training
   > The training process is define in train.py.
   Example to train autoencoder:
   ```bash
   net = train(train_loader, net, args, logger)
   ```
* MFA tool
  * The tool for source separation by modulating latent code of autoencoder.
  Example for declaring MFA object:
  ```bash
   mfa = MFA.MFA_source_separation(net, FFT_dict=FFT_dict, args=args)
   ```
### Requirements
MFA DAE is test using torch 1.7.0 with CUDA 10.1.
* librosa             0.8.0. 
* torch               1.7.0
* torchvision         0.8.2
* sklearn             0.0
* numpy               1.19.5
* scipy               1.6.0

### How to run
Example script to reproduce the training and evaluation procedures discussed in the paper is located in ./scripts.
```bash
  $sh ./scripts/example_MFA_.sh
```

The results will produce in default path ./log/default which can change by args.logdir.
### Example Data
There is an example heart-lung sound 0_0.wav in ./src/dataset/ folder.
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
