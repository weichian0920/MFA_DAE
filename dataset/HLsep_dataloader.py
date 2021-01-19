import sys
sys.path.append('../')
from torch.utils.data import DataLoader, Dataset
from utils.signalprocess import wav2lps, lps2wav, wav_read
import numpy as np
import scipy.io.wavfile as wav


class HL_dataset(Dataset):

    
    def __init__(self, data_path_list, FFT_dict, args):

        self.data_path_list = data_path_list
        self.FFT_dict = FFT_dict
        self.args  = args
        for filepath in self.data_path_list:
            if args.data_feature=="lps":
                spec, phase, mean, std = wav2lps(filepath, self.FFT_dict['FFTSize'],  self.FFT_dict['Hop_length'],  self.FFT_dict['Win_length'],  self.FFT_dict['normalize'])
                if args.model_type=="DAE_C":
                    self.samples = np.reshape((spec.T), (-1,1,1,int(self.FFT_dict['FFTSize']/2+1)))[:,:,:,self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1]]
                else:
                    self.samples = spec.T[:,self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1]]

            else:
                y = wav_read(filepath)
                self.samples = np.reshape(y, (-1,1,1,y.shape[0]))


    def __getitem__(self, index):

        return self.samples[index]


    def __len__(self):

        return len(self.samples)


def hl_dataloader(data_path_list, batch_size=311, shuffle=False, num_workers=1, pin_memory=True, FFT_dict=None, args=None):

    hl_dataset = HL_dataset(data_path_list, FFT_dict, args)
    hl_dataloader = DataLoader(hl_dataset, batch_size = batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return hl_dataloader

def val_dataloader(filepath, FFT_dict):

    lps, phase, mean, std = wav2lps(filepath, FFT_dict['FFTSize'], FFT_dict['Hop_length'], FFT_dict['Win_length'], FFT_dict['normalize'])

    return np.array(lps), np.array(phase), mean, std
