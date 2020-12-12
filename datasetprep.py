import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from hparams import Hparams
from tokenizers import RPT_Tokenizer


def load_data(hparams, split='|'):
    with open(hparams.transcriptpath, 'r', encoding='utf-8') as f:
        wavname_and_transcripts = [line.strip().split(split) for line in f]
    return wavname_and_transcripts


def data_split(wavname_and_transcripts, hparams):
    spl = int(len(wavname_and_transcripts) * hparams.trainval_split)
    traindata = wavname_and_transcripts[:spl]
    testdata = wavname_and_transcripts[spl:]
    return traindata, testdata


def load_wav(wavpath, hparams):
    waveform, sample_rate = torchaudio.load(wavpath)
    if hparams.resample_wavs and (hparams.target_sample_rate != sample_rate):
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=hparams.target_sample_rate)(waveform)
        return waveform
    
    assert sample_rate == hparams.target_sample_rate, "wav sample rate of {}Hz does not match target sample rate of {}Hz! Try setting resample_wavs=True".format(sample_rate, hparams.target_sample_rate)

    return waveform


def wav_to_mel(waveform, hparams):
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=hparams.target_sample_rate, n_mels=hparams.n_mels,  hop_length=hparams.hop_length, pad=0)(waveform)
    #mel = mel.log2().squeeze() OLD PROCESS
    with torch.no_grad():
        mel = 20 * torch.log10(torch.clamp(mel, min=1e-5)) - 20
        mel = torch.clamp((mel + 100) / 100, 0.0, 1.0).squeeze()
    return mel


def max_mel_len(hparams):
    """
    WARNING: This will take a long time if wavs are resampled
    on load and mels are not loaded from disk.
    """
    set = []
    data = load_data(hparams)
    for i in range(len(data)):
        wavpath = hparams.wavfolder + "\\" + data[i][0]
        wav = load_wav(wavpath, hparams)
        mel = wav_to_mel(wav, hparams)
        print(mel.shape[1])
        set.append(mel.shape[1])
    return max(set)


class RPT_Dataset(Dataset):
    """
    Core Dataset class. Description to be updated.
    """
    def __init__(self, datalist, hparams):
        self.datalist = datalist
        self.hparams = hparams
        self.tokenizer = RPT_Tokenizer(hparams)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        wavpath = self.hparams.wavfolder + "\\" + self.datalist[idx][0]
        wav = load_wav(wavpath, hparams=self.hparams)
        mel = wav_to_mel(waveform=wav, hparams=self.hparams)

        text = self.datalist[idx][1]

        return text, mel


class RPT_Dataloader(DataLoader):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        pass



# biglist = load_data(filepath=hparams.transcriptpath)

# traindata, valdata = trainsplit(wavname_and_transcripts=biglist, hparams=hparams)

# testloader = PrimeDataset(datalist=traindata, hparams=hparams)

# mel, text = testloader[22]

# print(text)

# plt.figure()
# plt.imshow(mel.numpy(), cmap='Blues')