from numpy.core.function_base import _logspace_dispatcher
import torch
import torchaudio
import os
import soundfile
import librosa


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
    if hparams.allow_resample == True and hparams.target_sample_rate != sample_rate:
        print('mismatched sample rate of {} found in {}, resampling...'.format(sample_rate, wavpath))
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=hparams.target_sample_rate)(waveform)
        print('Done.')
    assert sample_rate == hparams.target_sample_rate, "wav sample rate of {}Hz does not match target sample rate of {}Hz! If this is an outlier, set allow_resample to True in hparams. If this happens for every file, run resample_dataset.".format(sample_rate, hparams.target_sample_rate)
    return waveform


def wav_to_mel(waveform, max_mel_len, hparams):
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


def resample_datset(hparams):
    """
    Function to resample the provided sound files into wavs with the targeted sr provided in hparams.
    A folder for the resampled wavs will be created in the parent directory of the provided wav folder.
    """
    files = os.listdir(hparams.wavfolder)
    pathmod = hparams.wavfolder.split(sep='\\')
    pathmod[-1] += "_{}".format(hparams.target_sample_rate)
    newfolder = "\\".join(pathmod)
    assert not os.path.exists(newfolder), "The folder {} already exists! Has the dataset already been converted to this sample rate?".format(pathmod[-1])
    os.mkdir(newfolder)

    for file in files:
        loadpath = hparams.wavfolder + "\\" + file
        savepath = newfolder + "\\" + file
        print("Resampling {}...".format(file))
        wav, sr = librosa.load(loadpath, sr=hparams.target_sample_rate, mono=True)
        subtype = soundfile.info(loadpath, verbose=False).subtype
        soundfile.write(savepath, wav, samplerate=sr, subtype=subtype)

    print("Done.")