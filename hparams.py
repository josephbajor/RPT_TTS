from dataclasses import dataclass

@dataclass
class Hparams:
    transcriptpath: str = r"D:\Repos\Datasets\TestP2\masterset.txt"
    wavfolder: str = r"D:\Repos\Datasets\TestP2\wavs_22050"
    hop_length: int = 256 #This is probably going to need to match the value of the neural vocoder hop (300)
    windows_size: int = 1000
    remove_symbols: bool = True
    custom_tokens: bool = True
    device: str = 'cuda'


    target_sample_rate: int = 22050
    allow_resample: bool = False
    n_mels: int = 80
    
    
    epochs: int = 3000
    batch_size: int = 32
    lr_inital: float = 1e-2
    trainval_split: float = 0.9


    embed_dim: int = 512
    padding_idx: int = 0
    embed_conv_layers: int = 5
    embed_conv_dim: int = 512
    embed_kernel_size: int = 5
    embed_dropout: int = 0.5