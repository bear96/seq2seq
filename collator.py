import torch
import torch.nn as nn

class Collater:
    def __init__(self, src_lang, trg_lang):
        self.src_lang = src_lang
        self.trg_lang = trg_lang

    def __call__(self, batch):
        src_tensors, trg_tensors = zip(*batch)
        src_tensors = nn.utils.rnn.pad_sequence(src_tensors, batch_first=True, padding_value=2)
        trg_tensors = nn.utils.rnn.pad_sequence(trg_tensors, batch_first=True, padding_value=2)
        return src_tensors, trg_tensors
