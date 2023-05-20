import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, n_head=4, d_qkv=32, dropout=0.1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_qkv = d_qkv

        self.w_q = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_k = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_v = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_o = nn.Parameter(torch.Tensor(n_head, d_qkv, d_model))
        nn.init.xavier_normal_(self.w_q)
        nn.init.xavier_normal_(self.w_k)
        nn.init.xavier_normal_(self.w_v)
        nn.init.xavier_normal_(self.w_o)
        
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q,K,V,mask):
        #Q, K dot product
        QK_dot = Q.matmul(K.transpose(2,3))
        attention_weights = QK_dot/math.sqrt(self.d_qkv)
        #Attention weights will have shape (b, n, l, l)
#         print(attention_weights.shape, mask.shape)
        
        #apply masking
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask, -1e9)
        attention_weights = F.softmax(attention_weights,dim = -1)
        attention_weights = self.dropout(attention_weights)
        #final matmul
        #V has shape (b,n,l,d)
        V = attention_weights.matmul(V) #(l,l)x(l,d)=(l,d)
        
        return V, attention_weights

  
    def forward(self, x, mask):
        """Runs the multi-head self-attention layer.

        Args:
          x: the input to the layer, a tensor of shape [batch size, length, d_model]
          mask: a mask for disallowing attention to padding tokens. 
        Returns:
          A single tensor containing the output from this layer
        """
        
        # Implementation tip: using torch.einsum will greatly simplify the code that
        # you need to write.

        #x shape: (b, l, d_model)
        #W shape: (n_head, d_model, d)
        #output should have shape: (b, n, l, d)
        #linear

        #torch.Size([16, 60, 256]) torch.Size([4, 256, 32])
        Q = torch.einsum('blm, nmd->bnld', x, self.w_q)
        K = torch.einsum('blm, nmd->bnld', x, self.w_k)
        V = torch.einsum('blm, nmd->bnld', x, self.w_v)
    
        #apply attention
        V, attention_weights = self.scaled_dot_product_attention(Q,K,V,mask)
        
        #concat
        #Output has shape (b,l,m)
        output = V.matmul(self.w_o).sum(dim=1)
        return output
        
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model = 256, d_ff=1024, dropout=0.1):
        super().__init__()

        self.dense1 = nn.Linear(d_model, d_ff)
        self.dense2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ff=1024,n_head=4, d_qkv=32,dropout=0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, n_head=n_head, d_qkv = d_qkv, dropout = dropout)
        self.norm1 = nn.LayerNorm(normalized_shape = d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.feedforward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.1)
        self.norm2 = nn.LayerNorm(normalized_shape = d_model)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, mask):
        prev = x
        x = self.multi_head_attention(x, mask)
        x = self.dropout1(x)
        x = self.norm1(x + prev)
        
        prev = x
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = self.norm2(x + prev)
        
        return x
        
        

