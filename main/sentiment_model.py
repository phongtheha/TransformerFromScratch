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
  def __init__(self, d_model = 256, output_dim = None, vocab_size = None, n_layers = 4, d_ff=1024,n_head=4, d_qkv=32,dropout=0.1):
    super().__init__()
    self.d_model = d_model
    self.output_dim = output_dim
    self.vocab_size = vocab_size
    self.d_ff = d_ff
    self.n_head = n_head
    self.d_qkv = d_qkv
    self.dropout = dropout
    self.n_layers = n_layers

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
    #Here, encoder will have shape (batch_size, seq_length, d_model) so we will need to do a mean pool
    x = torch.mean(x, dim = 1)
    x = self.logit(x)
    #After logit layer, it will have shape (batch_size, num_of_classes, or 1 in case of binary)
    return x

  def compute_loss(self, batch):
    logits = self.encode(batch)
    logits = logits.reshape((-1,))
    labels = batch['labels'].reshape((-1,)).float()
    # res = F.cross_entropy(logits, labels, ignore_index=-1, reduction='mean')
    res = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
    return res
  
  def get_validation_metrics(self, val_dataset, batch_size=8, mode = "accloss"):
    assert mode in ['acc', 'loss','accloss']
    data_loader = torch.utils.data.DataLoader(
      val_dataset, batch_size=batch_size, collate_fn=val_dataset.collate)
    self.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
      for batch in data_loader:
        logits = self.encode(batch)
        if 'acc' in mode:     
          out = F.sigmoid(logits)
          predicted_labels = torch.tensor([1 if i == True else 0 for i in out > 0.5], device=device)
          gold_labels = batch['labels']
          correct += (predicted_labels == gold_labels).sum().item()
          total += gold_labels.shape[0]

        if 'loss' in mode:
          logits = logits.reshape((-1,))
          labels = batch['labels'].reshape((-1,)).float()
          loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
          total_loss += loss.item()
  
    if mode == 'acc':
      return (correct / total, None)
    elif mode == 'loss':
      return (None, total_loss)
    else:
      return (correct / total, total_loss)
  
  def train_model(self, train_dataset, val_dataset, num_epochs, batch_size, model_file,
          learning_rate=8e-4):
    """Train the model and save its best checkpoint.
    """
    history = {
    'train_loss': [],
    'val_loss': [],
    'val_acc': [],
    'epochs': num_epochs
}
    #Data loader for training
    data_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate)
    #Use Adam Optimizer
    optimizer = torch.optim.Adam(
        self.parameters(),
        lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    #Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(data_loader),
        pct_start=0.02,  # Warm up for 2% of the total training time
        )
    best_acc = 0.0
    for epoch in tqdm.trange(num_epochs, desc="training", unit="epoch", leave = True, position = 0):
      with tqdm.tqdm(
          data_loader,
          desc="epoch {}".format(epoch + 1),
          unit="batch",
          total=len(data_loader),position=0, leave=True) as batch_iterator:
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
        val_acc, val_loss = self.get_validation_metrics(val_dataset=val_dataset, mode = "accloss")
        history["train_loss"].append(total_loss/i)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        batch_iterator.set_postfix(
            mean_loss=total_loss / i,
            validation_metric=val_acc)
        if val_acc > best_acc:
          print(
              "Obtained a new best validation metric (Val Acc) of {:.3f}, saving model "
              "checkpoint to {}...".format(val_acc, model_file))
          torch.save(self.state_dict(), model_file)
          best_acc = val_acc
      
    print("Reloading best model checkpoint from {}...".format(model_file))
    self.load_state_dict(torch.load(model_file))
    return history

  # def make_prediction(self, example):
  #     self.eval()
  #     with torch.no_grad():
  #       result = self(example)
  #     return result


      
