#/--coding:utf-8/
#/author:Ethan Wang/
import scipy
import scipy.io.wavfile as wav
import numpy as np
import librosa
from scipy.signal import butter, lfilter, freqs
import math

def Filter(data, sr = 16000):
    """
    filter
    """
    filterOrder = 5
    nyq = 0.5*sr
    b, a = butter(filterOrder, [1000/nyq, 6000/nyq], btype='band')
    x = lfilter(b, a, data, axis = 0)
    return x

def wav2lps(y, FFTSize = 512, Hop_length = 256, Win_length = 512, normalize = False, Center = False):
    """
    This function transfer signal from waveform to log power spectrum.
    """
    if (type(y)==str):
        ##y = wav filepath of y is signal
        sr, y = wav.read(y)
        y = y/32767.
    epsilon = np.finfo(float).eps
    D = librosa.stft(y, n_fft = FFTSize, hop_length = Hop_length, win_length = Win_length, window = scipy.signal.hamming, center = Center)

    D = D + epsilon
    Sxx = np.log10(abs(D)**2)
    phase = np.exp(1j * np.angle(D))
    mean = np.mean(Sxx, axis=1).reshape(-1,1)
    std = np.std(Sxx, axis=1, dtype = np.float32, ddof=1).reshape(-1,1)

    if(normalize ==True):
        Sxx = (Sxx - mean)/std
        return Sxx, phase, mean, std
    else:
        return Sxx, phase, 0, 1

def lps2wav(lps, phase, mean = 0, std = 1, FFTSize = 512, Hop_length = 256, Win_length = 512, Center = False):
    """
    This function inverse the log power spectrum to wavform.
    output dim representation: feature, frame
    """
    epsilon = np.finfo(float).eps

    spec = lps*std+mean
    spec = np.sqrt(10**lps)
    spec = np.multiply(spec, phase)

    #spec = spec-epsilon
    sig = librosa.istft(spec, hop_length = Hop_length, win_length = Win_length, window = scipy.signal.hamming, center = Center)
    sig = np.int16(sig*32767.)

    return sig

def mixSnr(y_c, y_n, snr):
    """
    function for mixing clean signal with noise signal in indicated SNR
    """
    #noise = np.sqrt(np.var(y_c)/(math.pow(10, (snr/10))))*y_n[from_point:to_point]/np.std(y_n[from_point:to_point])
    clean_pwr = np.sum(abs(y_c)**2)/y_c.shape[0]
    noise = np.array(y_n) - np.mean(y_n)
    noise_var = clean_pwr/(10**(snr/10))
    noise = np.sqrt(noise_var)*y_n/np.std(y_n)
    
    Noisy = y_c + noise
    Noise = np.array(noise)
    noise = np.int16((Noisy - y_c)*32767./3.)
    clean = np.int16(y_c*32767.)
    Noisy = np.int16(np.array(Noisy*32767./3.))

    return Noisy, noise, clean
