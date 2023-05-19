import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import time

from transformer_encoder import TransformerEncoder
from positional_encoding import AddPositionalEncoding
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
PAD_ID = 0

class Transformer_Model(nn.Module):
  def __init__(self, d_model = 256, output_dim = None, vocab_size = None,  d_ff=1024,n_head=4, d_qkv=32,dropout=0.1):
    super().__init__()
    self.d_model = d_model
    self.output_dim = output_dim
    self.vocab_size = vocab_size
    self.d_ff = d_ff
    self.n_head = n_head
    self.d_qkv = d_qkv
    self.dropout = dropout

    self.add_timing = AddPositionalEncoding(self.d_model)
    self.encoder = TransformerEncoder(self.d_model, self.d_ff, self.n_layers, self.n_head, self.d_qkv, self.dropout)
    
    self.embedding = nn.Embedding(self.vocab_size, self.d_model)
    self.logit = nn.Linear(self.d_model, self.output_dim)


  def encode(self, batch):
    """
    Args:
      batch: an input batch as a dictionary; the key 'ids' holds the vocab ids
        of the subword tokens in a tensor of size [batch_size, sequence_length]
    Returns:
      A single tensor containing logits for each subword token
        
    """
    x = batch['ids']
    mask = (x == PAD_ID).unsqueeze(-2).unsqueeze(-2)
    x = self.embedding(x)
    x = self.add_timing(x)
    x = self.encoder(x, mask)
    x = self.logit(x)
    return x

  def compute_loss(self, batch):
    logits = self.encode(batch)
    logits = logits.reshape((-1, logits.shape[-1]))
    labels = batch['labels'].reshape((-1,))
    res = F.cross_entropy(logits, labels, ignore_index=-1, reduction='mean')
    return res
  
  def get_validation_metric(self, val_dataset, batch_size=8):
    data_loader = torch.utils.data.DataLoader(
      val_dataset, batch_size=batch_size, collate_fn=val_dataset.collate)
    self.eval()
    correct = 0
    total = 0
    with torch.no_grad():
      for batch in data_loader:
        mask = (batch['labels'] != -1)
        predicted_labels = self.encode(batch).argmax(-1)
        predicted_labels = predicted_labels[mask]
        gold_labels = batch['labels'][mask]
        correct += (predicted_labels == gold_labels).sum().item()
        total += gold_labels.shape[0]
    return correct / total
  
  def train_model(self, train_dataset, val_dataset, num_epochs, batch_size, model_file,
          learning_rate=8e-4):
    """Train the model and save its best checkpoint.
    """
    data_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate)
    optimizer = torch.optim.Adam(
        self.parameters(),
        lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(data_loader),
        pct_start=0.02,  # Warm up for 2% of the total training time
        )
    best_metric = 0.0
    for epoch in tqdm.trange(num_epochs, desc="training", unit="epoch"):
      with tqdm.tqdm(
          data_loader,
          desc="epoch {}".format(epoch + 1),
          unit="batch",
          total=len(data_loader)) as batch_iterator:
        self.train()
        total_loss = 0.0
        for i, batch in enumerate(batch_iterator, start=1):
          optimizer.zero_grad()
          loss = self.compute_loss(batch)
          total_loss += loss.item()
          loss.backward()
          optimizer.step()
          scheduler.step()
          batch_iterator.set_postfix(mean_loss=total_loss / i)
        validation_metric = self.get_validation_metric(val_dataset=val_dataset)
        batch_iterator.set_postfix(
            mean_loss=total_loss / i,
            validation_metric=validation_metric)
        if validation_metric > best_metric:
          print(
              "Obtained a new best validation metric of {:.3f}, saving model "
              "checkpoint to {}...".format(validation_metric, model_file))
          torch.save(self.state_dict(), model_file)
          best_metric = validation_metric
    print("Reloading best model checkpoint from {}...".format(model_file))
    self.load_state_dict(torch.load(model_file))


      
