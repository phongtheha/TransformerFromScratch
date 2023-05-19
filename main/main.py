import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import tqdm
import utils
from sklearn.model_selection import train_test_split
from sentiment_model import Transformer_Model
from transformers import BertTokenizer

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
def main():
  loaded_data = utils.load_data("./data/IMDB Dataset.csv")
  train_data, test_data, train_labels, test_labels = train_test_split(loaded_data[0], loaded_data[1], test_size=0.2, random_state=42)
  train_dataset = utils.IMDBDataset(data = (train_data, train_labels), max_length = 250)
  test_dataset = utils.IMDBDataset(data = (test_data, test_labels), max_length=250)

  sentiment_model = Transformer_Model(d_model = 256, output_dim = 1, vocab_size = BertTokenizer.from_pretrained('bert-base-uncased').vocab_size,
                                      d_ff=1024,n_head=4, d_qkv=32,dropout=0.1).to(device)
  


  sentiment_model.train_model(train_dataset = train_dataset, val_dataset = test_dataset, num_epochs = 2, batch_size = 16, model_file = "./sentiment_model.pt",
          learning_rate=8e-4)


if __name__ == "__main__":
  
  main()


