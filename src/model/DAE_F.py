import numpy as np
import sys
sys.path.append('./')
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
    """This function construct Encoder by fully-connected units.
    Arguments:
        model_dict: each encoder layer's parameters, including input feature_dim, layer's neurons number, and specified activation fucntion.
    """
    def __init__(self, model_dict=None, padding="same", args=None, logger=None):
        super(Encoder, self).__init__()

        self.model_dict = model_dict
        self.feature_dim = self.model_dict['feature_range'][1] - self.model_dict['feature_range'][0]
        self.encoder_act = self.model_dict['encoder_act']
        self.encoder_layer = self.model_dict['encoder']

        self.padding = padding
        self.layers = self._make_layers()

    def _make_layers(self):

        layers = []
        in_planes = self.feature_dim
        
        for i in range(0, len(self.encoder_layer)):
            out_planes = self.encoder_layer[i]
            layer = nn.Linear(in_planes, out_planes)

            in_planes = out_planes
            layers.append(layer)
            layers.append(ACT(self.encoder_act))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class Decoder(nn.Module):
    ###
    def __init__(self, model_dict, padding="same", args=None, logger=None):
        super(Decoder, self).__init__()
        self.model_dict = model_dict
        self.input_dim = self.model_dict['encoder'][-1]
        self.decoder_act = self.model_dict['decoder_act']
        self.decoder_layer = self.model_dict['decoder']
        self.layers = self._make_layers()

    def _make_layers(self):
       
        layers = []
        in_planes = self.input_dim 

        for i in range(0, len(self.decoder_layer)):
            out_planes = self.decoder_layer[i]
            layer= nn.Linear(in_planes, out_planes)
            layers.append(layer)
            layers.append(ACT(self.decoder_act))
            in_planes = out_planes

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layers(x)

        return x


class autoencoder(nn.Module):

    #deep convolutional autoencoder

    def __init__(self, model_dict=None, padding="same", args=None, logger=None):
        super(autoencoder, self).__init__()
        if model_dict==None:
            self.model_dict = {
        "feature_range":[0, 257],
        "encoder":[1024, 512, 256, 32],
        "decoder":[256, 512, 1024, 257],
        "encoder_act":"relu",
        "decoder_act":"relu",
             }
        else:
            self.model_dict = model_dict
        
        logger('============model_dict==========')
        for k,v in self.model_dict.items():
            logger('{}: {}'.format(k,v))
        logger('================================')
        
        self.feature_dim = self.model_dict['feature_range'][1] - self.model_dict['feature_range'][0]
        self.encoder = Encoder(self.model_dict)
        #self.PC = PC()
        self.decoder = Decoder(self.model_dict)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
