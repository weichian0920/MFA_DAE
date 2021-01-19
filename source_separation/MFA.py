# -*- coding: utf-8 -*-
#Author:Wei-Chien Wang

import sys
sys.path.append('../')
import torch
import numpy as np
import librosa
import scipy
from utils.clustering_alg import K_MEANS,  NMF_clustering
import os
import scipy.io.wavfile as wav
from utils.signalprocess import lps2wav


class MFA_source_separation(object):

    """
    MFA analysis for unsupervised monoaural source separation.
        This function separate different sources by unsupervised manner depend on different source's periodicity properties.
    Arguments:
        model: deep autoencoder for source separation.
        source number: separated source number.
        clustering _alg: clustering algorithm for MFA analysis.
        wienner_mask: if True the output is mask by constructed ratio mask.
        FFT_dict: fouier transform parameters.
    """


    def __init__(self, model=None, FFT_dict=None, args=None):

        self.model = model
        self.source_num = args.source_num
        self.clustering_alg = args.clustering_alg
        self.wienner_mask = args.wienner_mask
        self.FFT_dict = FFT_dict
        self.args = args


    def FFT_(self, input):

        epsilon = np.finfo(float).eps
        frame_num = input.shape[1]
        encoding_shape = input.shape[0]
        FFT_result = np.zeros((encoding_shape, int(frame_num/2+1)))

        for i in range(0, encoding_shape):
            fft_r = librosa.stft(input[i,:], n_fft=frame_num, hop_length=frame_num+1, window=scipy.signal.hamming)
            fft_r = fft_r+ epsilon
            FFT_r = abs(fft_r)**2
            FFT_result[i] = np.reshape(FFT_r, (-1,))

        return FFT_result
    

    def freq_modulation(self, source_idx, label, encoded_img):
        """
          This function use to modulate latent code.
          Arguments:
            source_idx: source no. 
            Encoded_img: latent coded matrix. Each dimension represnet as (encoding_shape, frame_num)
            label: latent neuron label.
        """
        frame_num = encoded_img.shape[0]
        encoding_shape = encoded_img.shape[1]
        # minimun value of latent unit which use to deactivate neuron.
        min_value = torch.min(encoded_img)

        for k in range(0, encoding_shape):
            if(label[k] != source_idx):
                # (encoding_shape,frame_num)
                encoded_img[:,k] = min_value
        return encoded_img


    def MFA(self, input, source_num=2):
        """
          This function attempt to find each latent neuron is activated by which source by clustering(K_MEANS of sparsity NMF clustering).
          Note: Each dimension of input represent as (frame number, encoded neuron's number).
        """
        encoded_dim  = input.shape[1]
        # Period clustering
        fft_bottleneck = self.FFT_(input.T)#(fft(encoded, frame_num))

        if self.clustering_alg == "K_MEAMS":
            k_labels, k_centers = K_MEANS.create_cluster(np.array(fft_bottleneck[:, 2:50]), source_num)
        else:
            _, _, k_labels, _ = NMF_clustering.basis_exchange(np.array(fft_bottleneck[:, 3:]).T, np.array(fft_bottleneck[:, 3:]), np.array([source_num]), segment_width = 1)

        return k_labels


    def source_separation(self, input, phase, mean, std, filedir, filename):
        """
          main function for blind source separation.
        """
 
        feature_dim = self.FFT_dict['frequency_bins'][1]-self.FFT_dict['frequency_bins'][0]

        if self.args.model_type == "DAE_C":
            x = np.reshape((input.T), (-1, 1, 1, int(self.FFT_dict['FFTSize']/2+1)))[:,:,:,self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1]]
        else:
            x = input.T[:, self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1]]
        x = torch.tensor(x).float().cuda()
        sources = torch.unsqueeze(torch.zeros_like(x), 0)
        # Encode input
        latent_code = self.model.encoder(x)
        # MFA analysis for identifying latent neurons 's label
        label = self.MFA(latent_code.cpu().detach().numpy())
        # Reconstruct input
        sources[0] = self.model.decoder(latent_code)
        # Discriminate latent code for different sources.
        for source_idx in range(0, self.source_num):
            y_s = self.freq_modulation(source_idx, label, latent_code)
            sources = torch.cat((sources, torch.unsqueeze(self.model.decoder(y_s), 0)), 0)
        sources = torch.squeeze(sources).permute(0, 2, 1).detach().cpu().numpy()
        # Source separation
        for source_idx in range(0, self.source_num+1):
            sources[source_idx,:,:] = np.sqrt(10**((sources[source_idx,:,:]*std[self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1],:])+mean[self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1],:]))

        input = np.sqrt(10**(input*std+mean))
        for source_idx in range(0, self.source_num+1):
            Result = np.array(input)
            if(self.wienner_mask==True):
                if source_idx == 0:# Reconstruct original signal
                    Result[self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1],:] = sources[0,:,:]
                else:
                    Result[self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1],:] =  2*(sources[source_idx,:,:]/(np.sum(sources[1:,:,:], axis = 0)))*sources[0,:,:]
            else:#Wienner_mask==False
                Result[self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1],:] = np.array(sources[source_idx,:,:])
            R = np.multiply(Result, phase)
            result = librosa.istft(R, hop_length=self.FFT_dict['Hop_length'], win_length=self.FFT_dict['Win_length'], window=scipy.signal.hamming, center=False)
            result = np.int16(result*32768)
            if source_idx==0:
                result_path = "{0}reconstruct/".format(filedir)
            else:
                result_path = "{0}source{1}/".format(filedir, source_idx)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            wav.write("{0}{1}.wav".format(result_path,  filename), self.FFT_dict['sr'], result)
