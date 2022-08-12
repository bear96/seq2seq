import pandas as pd
import numpy as np
from typing import Tuple
import sys
from transformers import RobertaTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from sklearn import metrics, model_selection
from matplotlib import pyplot as plt 
from tqdm import tqdm 
from sklearn import metrics, model_selection

from transformers import RobertaTokenizer

SOS_token = 0
EOS_token = 1
PAD_token = 2

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
class PolyLang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS",2: "PAD"}
        self.n_words = 3  

    def addSentence(self, sentence):
        for word in tokenizer.tokenize(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def readLangs(lang1,lang2, file_path):
    print("Reading lines...")

    # Read the file and split into lines
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])

    Factor_vocab = PolyLang(lang1)
    Expansions_vocab =  PolyLang(lang2)

    return Factor_vocab, Expansions_vocab, factors,expansions

from tqdm import tqdm
MAX_LENGTH = 30

def preparevocab(lang1, lang2,file_path):
    f_vocab,expans_vocab, factors,expansions = readLangs(lang1,lang2, file_path)
    print("Read %s sentence pairs" % len(factors))
    print("Building Polynomial vocabulary...")
    for i in tqdm(range(len(factors))):
        f_vocab.addSentence(factors[i])
        expans_vocab.addSentence(expansions[i])
    print("Counted words:")
    print(f_vocab.name, f_vocab.n_words)
    print(expans_vocab.name, expans_vocab.n_words)
    return f_vocab, expans_vocab,factors,expansions


f_vocab, expans_vocab,factors,expansions = preparevocab('Factors','Expansions', 'train.txt')

##Split data into train test here
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in tokenizer.tokenize(sentence)]


def tensorFromSentence(lang, sentence,device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(lang1,lang2, factors, expansions):
    input_tensor = tensorFromSentence(lang1, factors)
    target_tensor = tensorFromSentence(lang2, expansions)
    return (input_tensor, target_tensor)

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class InputFeatures(object):
  #to process data into ids and mask for the custom dataloader
  def __init__(self,input_ids,target_ids):
    self.input_ids = input_ids
    self.target_ids =target_ids


def convert_examples_to_features(factors,expansions,lang1,lang2,block_size=30):
    #target
    exps = expansions
    exps_tokens= indexesFromSentence(lang2,expansions)[:block_size-2]
    target_ids =[0]+exps_tokens+[1]
    padding_length = block_size - len(target_ids)
    target_ids+=[2]*padding_length
 
    #source
    facts = factors
    facts_tokens=indexesFromSentence(lang1,factors)[:block_size-2]
    source_ids =  [0]+facts_tokens+[1]
    padding_length = block_size - len(source_ids)
    source_ids+=[2]*padding_length

    return InputFeatures(source_ids,target_ids)

class PolyData(Dataset):
    def __init__(self, lang1,lang2, factors,expansions):
        self.examples = []
        for i in tqdm(range(len(factors)),desc = "Processing dataset..."):
          fact,exp = factors[i],expansions[i]
          self.examples.append(convert_examples_to_features(fact,exp,lang1,lang2,block_size=30))
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, indx):       
        return torch.tensor(self.examples[indx].input_ids),torch.tensor(self.examples[indx].target_ids)

dataset = pd.DataFrame({'factors': factors,'expansions':expansions})
train_data,val_data = model_selection.train_test_split(dataset,train_size=0.8,random_state=44)
test_data,val_data = model_selection.train_test_split(val_data,train_size=0.25,random_state=44)

train_dataset = PolyData(f_vocab,expans_vocab,train_data.factors.to_numpy(),train_data.expansions.to_numpy())
test_dataset = PolyData(f_vocab,expans_vocab,test_data.factors.to_numpy(),test_data.expansions.to_numpy())
val_dataset = PolyData(f_vocab,expans_vocab,val_data.factors.to_numpy(),val_data.expansions.to_numpy())

# import torch
# import torch.nn as nn


# class Transformer(nn.Module):
#     def __init__(
#         self,
#         embedding_size,
#         src_vocab_size,
#         trg_vocab_size,
#         src_pad_idx,
#         max_len,
#         device,
#         num_heads=8,
#         num_encoder_layers=6,
#         num_decoder_layers=6,
#         forward_expansion=2048,
#         dropout=0.1,

#     ):
#         super(Transformer, self).__init__()
#         self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
#         self.src_position_embedding = nn.Embedding(max_len, embedding_size)
#         self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
#         self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

#         self.device = device
#         self.transformer = nn.Transformer(
#             embedding_size,
#             num_heads,
#             num_encoder_layers,
#             num_decoder_layers,
#             forward_expansion,
#             dropout,
#         )
#         self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
#         self.dropout = nn.Dropout(dropout)
#         self.src_pad_idx = src_pad_idx

#     def make_src_mask(self, src):
#         src_mask = src.transpose(0, 1) == self.src_pad_idx

#         # (N, src_len)
#         return src_mask.to(self.device)

#     def forward(self, src, trg):
#         src_seq_length, N = src.shape
#         trg_seq_length, N = trg.shape

#         src_positions = (
#             torch.arange(0, src_seq_length)
#             .unsqueeze(1)
#             .expand(src_seq_length, N)
#             .to(self.device)
#         )

#         trg_positions = (
#             torch.arange(0, trg_seq_length)
#             .unsqueeze(1)
#             .expand(trg_seq_length, N)
#             .to(self.device)
#         )

#         embed_src = self.dropout(
#             (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
#         )
#         embed_trg = self.dropout(
#             (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
#         )

#         src_padding_mask = self.make_src_mask(src)
#         trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
#             self.device
#         )

#         out = self.transformer(
#             embed_src,
#             embed_trg,
#             src_key_padding_mask=src_padding_mask,
#             tgt_mask=trg_mask,
#         )
#         out = self.fc_out(out)
#         return out

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

print(device)

#model = Transformer(embedding_size =256,src_vocab_size = f_vocab.n_words,trg_vocab_size = expans_vocab.n_words,src_pad_idx=2,max_len=30,device=device,num_encoder_layers=3,num_decoder_layers=3,forward_expansion=256).to(device)

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
    
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == 2).transpose(0, 1)
    tgt_padding_mask = (tgt == 2).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

torch.manual_seed(0)

SRC_VOCAB_SIZE = f_vocab.n_words
TGT_VOCAB_SIZE = expans_vocab.n_words
EMB_SIZE = 256
NHEAD = 8
FFN_HID_DIM = 256
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
print("Total trainable params: ",pytorch_total_params)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=2)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.00005)
train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=32,shuffle=False)

for i in range(10):   
    transformer.train()
    tr_loss = 0.0
    val_loss =0.0
    for batch in tqdm(train_dataloader):
        src = batch[0].T.to(device)
        tgt = batch[1].T.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = transformer(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1)

        optimizer.zero_grad()
        tgt_out = tgt[1:, :]
        #print(logits.reshape(-1, logits.shape[-1]).shape,tgt_out.reshape(-1).shape)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        tr_loss += loss.item()

    transformer.eval()
    for batch in tqdm(val_dataloader):
        src = batch[0].T.to(device)
        tgt = batch[1].T.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = transformer(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        val_loss += loss.item()

    print("Epoch: {}, Train Loss: {}, Val Loss: {}".format(i+1,tr_loss/len(train_dataloader),val_loss/len(val_dataloader)))

test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=False)
transformer.eval()
pred_expansions = []
for batch in tqdm(test_dataloader):
    num_tokens = batch[0].shape[1]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    src = batch[0].T.to(device)
    src_mask = src_mask.to(device)

    memory = transformer.encode(src, src_mask)
    output_ids = []
    ys = torch.ones(1, 1).fill_(0).type(torch.long).to(device)
    for i in range(30-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)
        out = transformer.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = transformer.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == 1:
            break
        output_ids.append(next_word)
    pred_expansions.append(output_ids)

expansions[32]

for idx in pred_expansions[32]:
  word = expans_vocab.index2word[idx]
  print(word)
  s = ""
  s.join(word)
print(s)

