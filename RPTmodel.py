import torch
import torch.nn as nn
import datasetprep
import tokenizers
import Embedding
from hparams import Hparams

hparams = Hparams()

class RPT_Model(nn.Module):
    """
    placeholder
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
    pass