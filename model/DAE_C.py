# -*- coding: utf-8 -*-
#Author:Wei-Chien Wang

import numpy as np
import os
import torch
import torch.nn as nn
import math


def ACT(act_f):
   if(act_f=="relu"):
       return torch.nn.ReLU()
   elif(act_f=="tanh"):
       return torch.nn.Tanh()
   elif(act_f=="relu6"):
       return nn.ReLU6()
   elif(act_f=="sigmoid"):
       return nn.Sigmoid()
   elif(act_f=="LeakyReLU"):
       return nn.LeakyReLU()
   else:
       print("Doesn't support {0} activation function".format(act_f))


class Encoder(nn.Module):


    def __init__(self, model_dict=None, padding="same", args=None, logger=None):
        super(Encoder, self).__init__()
        self.model_dict = model_dict
        self.padding = padding
        self.feature_dim = self.model_dict['frequency_bins'][1] - self.model_dict['frequency_bins'][0]
        self.encoder_act = self.model_dict['encoder_act']
        self.encoder_layer = self.model_dict['encoder']
        self.encoder_filter = self.model_dict['encoder_filter']
        self.dense_layer = self.model_dict['dense']
        self.conv_layers = self._make_layers()

        if(len(self.dense_layer)!=0):
            if(self.padding=="same"):
                input_dim = self.encoder_layer[-1]*self.feature_dim
                self.dense_layer = nn.Linear(input_dim, self.dense_layer[0])
                self.dense_act = ACT(self.encoder_act)
            else:
                s_ = np.sum(np.array(self.encoder_filter), axis = 1) - np.array(self.encoder_filter).shape[1]
                input_dim = self.encoder_layer[-1]*(self.feature_dim - s_)
                self.dense_layer = nn.Linear(input_dim, self.dense_layer[0])
                self.dense_act = ACT(self.encoder_act)


    def _make_layers(self):

        layers = []
        in_channels = 1

        for i in range(0, len(self.encoder_layer)):
            out_channels = self.encoder_layer[i]
            pad_layer = nn.ZeroPad2d(padding=(0, self.encoder_filter[i][1]-1, 0, self.encoder_filter[i][0]-1))
            if(self.padding=="same"):
                layers.append(pad_layer)
            encoder_layer = nn.Conv2d(in_channels, out_channels, kernel_size = (self.encoder_filter[i][0], self.encoder_filter[i][1]), stride = (1,1), padding = 0, bias = True)

            in_channels = out_channels
            layers.append(encoder_layer)
            layers.append(ACT(self.encoder_act))

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.conv_layers(x)
        x = torch.reshape(x, (x.shape[0], -1))
        if(len(self.model_dict["dense"])!=0):
            x = self.dense_layer(x)
            x = self.dense_act(x)

        return x


class Decoder(nn.Module):
    ###
    def __init__(self, model_dict, padding="same", args=None, logger=None):
        super(Decoder, self).__init__()
        self.model_dict = model_dict
        self.dense_l = self.model_dict['dense']
        self.feature_dim = self.model_dict['frequency_bins'][1] - self.model_dict['frequency_bins'][0]
        self.decoder_act = self.model_dict['decoder_act']
        self.decoder_layer = self.model_dict['decoder']
        self.decoder_filter = self.model_dict['decoder_filter']
        self.conv_layers = self._make_layers()
        if(len(self.dense_l)!=0):
            self.dense_layer = nn.Linear(self.dense_l[0], self.feature_dim)
            self.dense_act = ACT(self.decoder_act)


    def _make_layers(self):
       
        layers = []

        if len(self.dense_l)==0:
            in_channels = self.decoder_layer[0]
        else: 
            in_channels = 1

        for i in range(0, len(self.decoder_layer)):
            out_channels = self.decoder_layer[i]
            if i == (len(self.decoder_layer)-1):
                decoder_layer = nn.Conv2d(in_channels, out_channels, kernel_size = (self.decoder_filter[i][0], self.decoder_filter[i][1]), stride = (1,1), padding = 0, bias = True)
                layers.append(decoder_layer)
            else:
                decoder_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = (self.decoder_filter[i][0], self.decoder_filter[i][1]), stride=(1,1), padding = (0,1))
                layers.append(decoder_layer)
                layers.append(ACT(self.decoder_act))
            in_channels = out_channels
        return nn.Sequential(*layers)


    def forward(self, x):

        if len(self.dense_l) == 0: 
            x = torch.reshape(x, (-1, self.decoder_layer[0], 1, self.feature_dim))
        else:
            x = self.dense_layer(x)
            x = self.dense_act(x)
            x = torch.reshape(x, (-1, 1, 1, self.feature_dim))
        x = self.conv_layers(x)

        return x


class autoencoder(nn.Module):

    """
    # deep convolutional autoencoder
    """

    def __init__(self, model_dict=None, padding="same", args=None, logger=None):
        super(autoencoder, self).__init__()
        if model_dict==None:
            self.model_dict = {
                "frequency_bins": [0, 257],
                "encoder": [1024, 512, 256, 32],
                "decoder": [32, 256, 512, 1024, 1],
                "encoder_filter": [[1, 3],[1, 3],[1, 3],[1, 3]],
                "decoder_filter": [[1, 3],[1, 3],[1, 3],[1, 3],[1, 1]],
                "encoder_act": "relu",
                "decoder_act": "relu",
                "dense": [16],
                }
        else:
            self.model_dict = model_dict
        self.feature_dim = self.model_dict['frequency_bins'][1] - self.model_dict['frequency_bins'][0]
        self.encoder = Encoder(self.model_dict)
        self.decoder = Decoder(self.model_dict)


    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

if __name__=="__main__":
    #####test phasea
    x = torch.tensor(np.ones((10, 1, 1, 257))).float()
    model_dict = {
        "frequency_bins":[0, 257],
        "encoder":[1024, 512, 256, 32],
        "decoder":[32, 256, 512, 1024, 1],
        "encoder_filter":[[1,3],[1,3],[1,3],[1,3]],
        "decoder_filter":[[1,3],[1,3],[1,3],[1,3],[1,1]],
        "encoder_act":"relu",
        "decoder_act":"relu",
        "dense":[16],
        }
    
    encoder = Encoder(model_dict)
    decoder = Decoder(model_dict)
    output = encoder(x)
    print(output.shape)
    x_ = decoder(output)
    print(x_.shape)
    
    model = autoencoder()
    y = model(x)
    print("output.shape", y.shape)
    l = model.encoder(x)
    print("l.shape", l.shape)
    y = model.decoder(l)
    print("y.shape", y.shape)

