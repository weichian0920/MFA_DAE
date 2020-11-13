#/--coding:utf-8/
#/author:Ethan Wang/

import torch
import numpy as np
import librosa
import scipy
from utils.clustering_alg import K_MEANS,  NMF_clustering
import os
import scipy.io.wavfile as wav
from utils.signalprocess import lps2wav

class MFA_source_separation(object):
    """MFA analysis for unsupervised monoaural source separation.
    this function separate different sources in unsupervised manner depend on their periodicity properties.
    # Arguments
      source number: indicate separated source number.
      clustering _alg: clustering algorithm for MFA analysis.
      wienner_mask: the output would be mask by constructed ratio mask.
    """
    def __init__(self, model=None, args=None):

        self.model = model
        self.clustering_alg = args.clustering_alg
        self.source_num = args.source_num
        self.wienner_mask = args.wienner_mask

    def FFT_(self, input):

        epsilon = np.finfo(float).eps

        frame_num = input.shape[1]
        encoding_shape = input.shape[0]
    
        FFT_result = np.zeros((encoding_shape, int(frame_num/2+1)))
        for i in range(0, encoding_shape):
            fft_r = librosa.stft(input[i,:], n_fft=frame_num, hop_length = frame_num+1, window = scipy.signal.hamming)
            fft_r = fft_r+ epsilon
            FFT_r = abs(fft_r)**2
            FFT_result[i] = np.reshape(FFT_r, (-1,))

        return FFT_result
    
    def freq_modulation(self, source_idx, label, encoded_img):

        #Encoded_img : latent coded matrix
        #latent label : latent neuron label
        #min_value : minimun value of latent unit which use to deactivate neuron
        frame_num = encoded_img.shape[0]# encoded_img = (encoding_shape, frame_num)
        encoding_shape = encoded_img.shape[1]# encoded_img = (encoding_shape, frame_num)
        min_value = torch.min(encoded_img)

        for k in range(0, encoding_shape):
            if(label[k] != source_idx):
                encoded_img[:,k] = min_value#(encoding_shape,frame_num)

        return encoded_img

    def MFA(self, input, source_num = 2):
        """
        The input x dim represent is (frame number, encoded neuron's number)
        """
        encoded_dim  = input.shape[1]
        ###############period clustering
        fft_bottleneck = self.FFT_(input.T)#(fft(encoded, frame_num))

        if self.clustering_alg=="K_MEAMS":
            k_labels, k_centers = K_MEANS.create_cluster(np.array(fft_bottleneck[:, 2:50]), source_num)
        else:
            _, _, k_labels, _ = NMF_clustering.basis_exchange(np.array(fft_bottleneck[:, 3:]).T, np.array(fft_bottleneck[:, 3:]), np.array([source_num]), segment_width = 1)

        return k_labels

    def source_separation(self, input, phase, mean = 0, std = 1, filedir="../test/result/", filename="test"):

        input = torch.tensor(input).cuda()
        sources = torch.unsqueeze(torch.zeros_like(input), 0)
        latent_code = self.model.encoder(input)
        #MFA analysis for identifying latent neurons 's label
        label = self.MFA(latent_code.cpu().detach().numpy())
        #reconstruct input
        sources[0] = self.model.decoder(latent_code)
        #generate the latent code and decode the separated latent code
        for source_idx in range(0, self.source_num):
            y_s = self.freq_modulation(source_idx, label, latent_code)
            sources = torch.cat((sources, torch.unsqueeze(self.model.decoder(y_s), 0)), 0)
        sources = torch.squeeze(sources).permute(0,2,1).detach().cpu().numpy()
        
        #
        for source_idx in range(0, self.source_num+1):
            if(self.wienner_mask==True):
                if source_idx==0:#reconstruct original signal
                    Result = sources[0,:,:]*std+mean
                else:
                    Result = ((sources[source_idx,:,:]*std+mean)/(np.sum(sources[1:,:,:], axis = 0)*std+mean)*sources[0,:,:]*std+mean)
            else:#wienner_mask==False
                Result = np.array(sources[source_idx,:,:])*std+mean
            Result = np.sqrt(10**(Result))
            R = np.multiply(Result, phase)
            result = librosa.istft(R, hop_length = 256, win_length = 512, window=scipy.signal.hamming, center = False)
            result = np.int16(result*32768)
            if source_idx==0:
                result_path = "{0}reconstruct/".format(filedir)
            else:
                result_path = "{0}source{1}/".format(filedir, source_idx)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            wav.write("{0}{1}.wav".format(result_path,  filename), 8000, result)

    
