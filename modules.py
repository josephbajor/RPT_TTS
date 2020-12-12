#Performer module adapted from 

#General inspiration from Microsoft's FastSpeech2, Google's Tacotron2, and NVIDIA's Flowtron

# WHAT YOU ARE CURRENTLY DOING:
# refactoring reformer tts code to see if it works
# in the context of the dataloader and to get a better grasp of
# the data flow through the system and the way it is processed
# After that works, will move onto constructing the actual network
# P.S. look to the wavegrad code for tensorboard implementation examples


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hparams import Hparams


class Encoder(nn.module):
    pass

class Decoder(nn.module):
    pass




class Rapture(nn.module):
    pass