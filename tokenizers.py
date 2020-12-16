from unidecode import unidecode
import re
import torch
import torch.nn.functional as F
from preprocessing import load_data


def build_char_mapping_idx(charset):
    mapping_idx = {char:idx for idx, char in enumerate(charset)}
    return mapping_idx


def char_to_idx(text, mapping_idx):
    text = [mapping_idx[i] for i in text]
    return text


def text_cleaner(text: str, remove_symbols = True, custom_tokens = True):
    text = unidecode(text)
    
    if remove_symbols:
        text = re.sub(r'[\(\)\[\]\"#]+', '', text)
        text = re.sub("\;", ",", text)
        text = re.sub("\:", ",", text)
        text = re.sub("\-", " ", text)
        text = re.sub("\&", "and", text)

    if not custom_tokens:
        text = re.sub(r'[\*\<\>]+', '', text)
        text = re.sub(r'\.+', '\.', text)
        text = re.sub(r'\.{5}?', '.', text)
        text = re.sub(r'\.{3}?', '.', text)

    return text
    

def token_list(raw_data):
    """
    Scans all transcripts in the dataset and compiles a list of all unique
    tokens that appear, appended with custom tokens (if enabled).

    Args:
        raw_data: takes raw data dump from the load_data function

    Returns:
        list of unique tokens used in the set, list of custom tokens used

    """
    charset = []

    for i in range(len(raw_data)):
        text = raw_data[i][1]
        text = text_cleaner(text)
        text = tokenize_chars(text)
        for x in text:
            if x not in charset:
                charset.append(x)
    charset.sort()
    return charset


def dataset_max_len(raw_data):
    set = []
    for i in range(len(raw_data)):
        text = raw_data[i][1]
        text = text_cleaner(text)
        text = tokenize_chars(text)
        set.append(len(text))
    return max(set)


def tokenize_chars(text:str):
    """
    Breaks the input sequence into characters while extracting custom tags.
    Appends the sequence with an EOS token.
    """
    #tidx = re.sub(r'<.*?>', r'#', text)
    #token_idxr = re.compile(r'\*.*?\*')
    #tidx = re.sub(r'\*.*?\*', r'#', text)
    text = re.sub(r'\.{5}?', '*voc*', text)
    text = re.sub(r'\.{3}?', '*elp*', text)
    text = text.upper()
    #include line breaks in conversion
    #text = re.findall(r'\*[^*]*\*|.', text, re.S)
    text = re.findall(r'\*[^*]*\*|.', text)
    text = ['<SPC>' if i == ' ' else i for i in text]
    text.append('<EOS>')

    return text


def tokenize_phones():
    raise NotImplementedError




class RPT_Tokenizer():
    """
    Takes an input string and breaks it into tokens

    This system supports the use of custom style tokens. Transcriptions
    should provide token names inline enclosed by * or <>
    Example: *sigh*, <laugh>
    """
    def __init__(self, hparams):
        self.hparams = hparams
        self.charset = token_list(load_data(hparams))
        self.max_len = dataset_max_len(load_data(hparams))
        self.mapping_idx = build_char_mapping_idx(self.charset)

    def tokenize(self, text):
        text = text_cleaner(text, self.hparams.remove_symbols, self.hparams.custom_tokens)
        text = tokenize_chars(text)
        text = char_to_idx(text, self.mapping_idx)
        pad_len = self.max_len - len(text)
        text = torch.LongTensor(text)
        text = F.pad(text, (0, (pad_len)), mode='constant')
        return text