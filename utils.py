import os
import sys
import torchaudio
import torch


def pppdata_cleaner(filepath, outputpath):
   
    '''
    gets rid of the first directory listed for filepaths in dataset masterfiles
    '''
   
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        f.close()
    
    newfile = open(outputpath, 'w')
    for line in lines:
        line = line.split(r'/')[1]
        newfile.writelines(line)
        
    newfile.close()
    
    
class load_hparams():
    def __init__(self, filepath):
        self.filepath = filepath
        
    
def data_pathedit(filepath, outpath, wavfolder):
    pass


def MelGANdatasplit(path, tsplit=0.9):
    '''
    splits WAV data into 2 txt files.
    files will be put into WAV directory
    '''
    assert 0 <= tsplit <= 1, 'Tsplit must be less than 1!'
    
    files = os.listdir(path = path)
    spl = int(len(files) * tsplit)
    trainwavs = files[:spl]
    valwavs = files[spl:]
    
    trainpath = path + r'\trainwavs.txt'
    valpath = path + r'\valwavs.txt'
    
    trainfile = open(trainpath, 'w')
    for line in trainwavs:
        line = line + '\n'
        trainfile.writelines(line)
    trainfile.close()
        
    valfile = open(valpath, 'w')    
    for line in valwavs:
        valfile.writelines(line + '\n')
    valfile.close()


def batch_resample(path, sample_rate):
    '''
    WARNING: This results in an increased filesize
    the torchaudio save func has some issues
    '''
    files = os.listdir(path)
    for file in files:
        data, sr = torchaudio.load(path + '//' + file)
        data = torchaudio.transforms.Resample(sr, sample_rate)(data)
        torchaudio.save(path + '//' + file, data, sample_rate)
        print(f'resampling {file}...')
    print('Done.')


# def batch_resample(path, sample_rate):
#     files = os.listdir(path)
#     for file in files:
#         data = librosa.load(path=(path+'//'+file), sr=sample_rate)
#         soundfile.write(data=data, file=(path+'//'+file), samplerate=sample_rate, dtype='float32')
#         print(f'resampled {file}')
#     print('Done.')