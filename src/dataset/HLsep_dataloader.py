import sys
sys.path.append('../')
from torch.utils.data import DataLoader, Dataset
from utils.signalprocess import wav2lps, lps2wav
import numpy as np
import scipy.io.wavfile as wav


class HL_dataset(Dataset):
    def __init__(self, data_path_list, feature_dim, FFTSize, Hop_length, Win_length, normalize, args):

        self.data_path_list = data_path_list
        #self.name = name
        #self.mode = mode

        for filepath in self.data_path_list:
            print(normalize)
            spec, phase, mean, std = wav2lps(filepath, FFTSize, Hop_length, Win_length, normalize)
        if args.model_type=="DAE_C":
            self.samples = np.reshape((spec.T), (-1,1,1,257))
        else:
            self.samples = spec.T
            
    def __getitem__(self, index):

        #spec, _, mean, var = wav2lps(self.samples[index])
        return self.samples[index]

    def __len__(self):

        return len(self.samples)


def hl_dataloader(data_path_list, batch_size=311, shuffle=False, num_workers=1, pin_memory=True, feature_dim=257, FFTSize=512, Hop_length = 256, Win_length = 512, normalize=False, args=None):

    hl_dataset = HL_dataset(data_path_list, feature_dim, FFTSize, Hop_length, Win_length, normalize, args)
    hl_dataloader = DataLoader(hl_dataset, batch_size = batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return hl_dataloader

def val_dataloader(filepath, FFTSize=512, Hop_length = 256, Win_length = 512, normalize=False):
    lps, phase, mean, std = wav2lps(filepath, FFTSize, Hop_length, Win_length, normalize)
    return np.array(lps), np.array(phase), mean, std

if __name__=="__main__":
    """
    test phase
    """
    #data_path_list= ["/home/ethan/weichian/MFA_DAE/src_pytorch/dataset/0_0.wav", "/home/ethan/weichian/MFA_DAE/src_pytorch/dataset/0_0.wav"]
    data_path_list= ["/home/ethan/weichian/MFA_DAE/src_pytorch/dataset/0_0.wav"]
    train_loader = hl_dataloader(data_path_list)
    lps, phase, mean, std = val_dataloader(data_path_list[0])
    for i, (data) in enumerate(train_loader):
        print(data[1:2])

