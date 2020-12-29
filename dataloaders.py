import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tokenizers import RPT_Tokenizer
from preprocessing import data_split, load_data, load_wav, wav_to_mel, max_mel_len


def RPT_collate():
    pass


class RPT_Dataset(Dataset):
    """
    Core Dataset class. Description to be updated.
    """
    def __init__(self, hparams):
        self.hparams = hparams
        self.datalist = load_data(self.hparams)
        self.tokenizer = RPT_Tokenizer(self.hparams)
        #self.max_mel_len = max_mel_len(self.hparams)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        wavpath = self.hparams.wavfolder + "\\" + self.datalist[idx][0]
        wav = load_wav(wavpath, hparams=self.hparams)
        mel = wav_to_mel(waveform=wav, hparams=self.hparams)
        

        text = self.datalist[idx][1]
        tokens = self.tokenizer.tokenize(text)

        return tokens, mel



# biglist = load_data(filepath=hparams.transcriptpath)

# traindata, valdata = trainsplit(wavname_and_transcripts=biglist, hparams=hparams)

# testloader = PrimeDataset(datalist=traindata, hparams=hparams)

# mel, text = testloader[22]

# print(text)

# plt.figure()
# plt.imshow(mel.numpy(), cmap='Blues')