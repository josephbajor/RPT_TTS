
#Adapted from ESPNet2's tacotron encoder
#will list major changes made here:

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm1d


def embed_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, nn.init.calculate_gain('relu'))


class RPT_Embedding(nn.Module):
    def __init__(self, hparams, embed_idim):
        super().__init__()
        self.hparams = hparams
        self.embed_idim = embed_idim

        self.embed = nn.Embedding(embed_idim, hparams.embed_dim, padding_idx=hparams.padding_idx)

        self.conv_layers = nn.ModuleList()

        for layer in range(hparams.embed_conv_layers):
            conv_idim = hparams.embed_dim if layer == 0 else hparams.embed_conv_dim
            self.conv_layers += [
                nn.Sequential(
                    nn.Conv1d(
                        conv_idim,
                        hparams.embed_conv_dim,
                        hparams.embed_kernel_size,
                        stride=1,
                        padding=(hparams.embed_kernel_size -1) // 2,
                        bias=False
                    ),
                    #comment out batchnorm if attention issues persist
                    nn.BatchNorm1d(hparams.embed_conv_dim),
                    nn.ReLU(),
                    nn.Dropout(hparams.embed_dropout)
                )
            ]

        self.apply(embed_init)

    
    def forward(self, x):
        x = self.embed(x).transpose(0,1)
        for i in range (len(self.conv_layers)):
            x = self.conv_layers[i](x)
        return x.transpose(0,1)

    def inference(self, x):
        x = x.unsqueeze(0)
        return self.forward(x)[0][0]