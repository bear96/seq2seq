import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import process_dataset
from collator import Collater
from tqdm import tqdm
from sklearn import model_selection
import pickle

f_vocab, expans_vocab,factors,expansions = process_dataset.preparevocab('Factors','Expansions', 'train.txt')
dataset = pd.DataFrame({'factors': factors,'expansions':expansions})

train_data,val_data = model_selection.train_test_split(dataset,train_size=0.9,random_state=44)
test_data,val_data = model_selection.train_test_split(val_data,train_size=0.2,random_state=44)

train_dataset = process_dataset.PolyData(f_vocab,expans_vocab,train_data.factors.to_numpy(),train_data.expansions.to_numpy())
val_dataset = process_dataset.PolyData(f_vocab,expans_vocab,val_data.factors.to_numpy(),val_data.expansions.to_numpy())

test_dataset = process_dataset.PolyData(f_vocab,expans_vocab,test_data.factors.to_numpy(),test_data.expansions.to_numpy())

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

print("Device: ",device)

SRC_VOCAB_SIZE = f_vocab.n_words
TGT_VOCAB_SIZE = expans_vocab.n_words
EMB_SIZE = 256
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
collate_fn = Collater(f_vocab, expans_vocab)

def train(train_dataset,val_dataset,SRC_VOCAB_SIZE,SRC_VOCAB_SIZE,EMB_SIZE,NHEAD,FFN_HID_DIM,BATCH_SIZE,NUM_ENCODER_LAYERS,NUM_DECODER_LAYERS,collate_fn):
  torch.manual_seed(0)
  transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

  pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
  print("Total trainable params: ",pytorch_total_params)

  for p in transformer.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)
  
  transformer = transformer.to(device)

# #checkpoint
# checkpoint = False

# if checkpoint:
#     print("Loading training checkpoint.\n")
#     transformer.load_state_dict(torch.load('transformer_model.pkl'))

  loss_fn = torch.nn.CrossEntropyLoss(ignore_index=2)
  optimizer = torch.optim.Adam(transformer.parameters(), lr=0.00005)
  
  train_dataloader = DataLoader(train_dataset,batch_size=128,shuffle=True,collate_fn=collate_fn)
  val_dataloader = DataLoader(val_dataset,batch_size=128,shuffle=False,collate_fn=collate_fn)
  train_loss = []
  valid_loss = []
  for i in range(1):
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
    for batch in val_dataloader:
      src = batch[0].T.to(device)
      tgt = batch[1].T.to(device)

      tgt_input = tgt[:-1, :]

      src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

      logits = transformer(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

      tgt_out = tgt[1:, :]
      loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
      val_loss += loss.item()

    if tr_loss/len(train_dataloader)<val_loss/len(val_dataloader):
        print("Model is overfitting...")
        break
    print("Epoch: {}, Train Loss: {}, Val Loss: {}".format(i+1,tr_loss/len(train_dataloader),val_loss/len(val_dataloader)))
    train_loss.append(tr_loss/len(train_dataloader))
    valid_loss.append(val_loss/len(val_dataloader))
    
  #training and validation loss per epoch
  plt.plot(train_loss,'k')
  plt.plot(valid_loss,'y')
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend(["Training Loss","Validation Loss"])
  plt.title("Loss vs Epoch")
  plt.savefig('Loss_vs_Epoch.png')

  PATH = "transformer_model.pkl"

  torch.save(transformer.state_dict(), PATH)
  
def test(test_dataset,SRC_VOCAB_SIZE,SRC_VOCAB_SIZE,EMB_SIZE,NHEAD,FFN_HID_DIM,BATCH_SIZE,NUM_ENCODER_LAYERS,NUM_DECODER_LAYERS,collate_fn):
  torch.manual_seed(0)
  transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
  transformer.load_state_dict(torch.load('transformer_model.pkl'))
  print("Testing...")
  test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=False,collate_fn=collate_fn)
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
    for i in range(29):
      memory = memory.to(device)
      tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)
      out = transformer.decode(ys, memory, tgt_mask)
      out = out.transpose(0, 1)
      prob = transformer.generator(out[:, -1])
      _, next_word = torch.max(prob, dim=1)
      next_word = next_word.item()

      ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
      if next_word == 1:
        break
        
    pred_expansions.append(ys.cpu().numpy().reshape(-1))
    
    preds = decode_token_ids(pred_expansions)
    return preds
 
def decode_token_ids(pred_expansions):
  preds = []
  for i in range(len(pred_expansions)):
    wrd = []
    for idx in pred_expansions[i]:
      if idx==0:
        pass
      elif idx==1:
        break
      else:
        wrd.append(expans_vocab.index2word[idx])
      word = "".join(wrd)
    preds.append(word)
  s,actuals = get_score(test_data,preds)
  print("Saving...")
  d = {'Score':c/len(preds),'Preds':preds,'Actuals': list(actuals)}
  with open('preds.pkl', 'wb') as f:
    pickle.dump(d, f)
  

def score(true_expansion: str, pred_expansion: str) -> int:
  """ the scoring function - this is how the model will be evaluated
  :param true_expansion: group truth string
  :param pred_expansion: predicted string
  :return:
  """
  return int(true_expansion == pred_expansion)
def get_score(test_data,preds):
  actuals = test_data.expansions.to_numpy()
  c=0
  for i in range(len(preds)):
    c+=score(actuals[i],preds[i])
  print("Score: ",c/len(preds))
  return c/len(preds),actuals


train()
test()

