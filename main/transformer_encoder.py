import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_blocks import EncoderLayer


class TransformerEncoder(nn.Module):
  def __init__(self, d_model=256, d_ff=1024, n_layers=4, n_head=4, d_qkv=32,
               dropout=0.1):
    super().__init__()
   
    self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, d_ff=d_ff,\
                                              n_head=n_head, d_qkv=d_qkv,dropout=dropout)\
                                            for layer in range(n_layers)])

  def forward(self, x, mask):
    """Runs the Transformer encoder.

    Args:
      x: the input to the Transformer, a tensor of shape
         [batch size, length, d_model]
      mask: a mask for disallowing attention to padding tokens. 
    Returns:
      A single tensor containing the output from the Transformer
    """

    for layer in self.layers:
        x = layer(x, mask)
    return x
   