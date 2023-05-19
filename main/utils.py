import string
import nltk
import re
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
tqdm.pandas()
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

class TextPreprocessor:
    def __init__(self, rmStopWords = False, lowercase = False, rmPuncs = False,
                 rmSpecialChars = False, rmLinks = False, rmHtmlTags = False,
                 rmExtraSpaces = False, rmNumbers = False, lemmatize = False):
        self.stopwords = set(stopwords.words('english'))
        self.rmStopWords = rmStopWords
        self.lowercase = lowercase
        self.rmPuncs= rmPuncs
        self.rmSpecialChars = rmSpecialChars
        self.rmLinks = rmLinks
        self.rmHtmlTags = rmHtmlTags
        self.rmExtraSpaces = rmExtraSpaces
        self.rmNumbers = rmNumbers
        self.lemmatize = lemmatize
        
    def remove_stopwords(self, text):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords]
        return ' '.join(filtered_words)
    def space_bt_punct(self, text):
        pattern = r'([.,!?-])'
        s = re.sub(pattern, r' \1 ', text)     
        s = re.sub(r'\s{2,}', ' ', s)       
        return s
    def make_lowercase(self, text):
        return text.lower()
    
    def remove_punctuation(self, text):
        translator = str.maketrans('', ' ', string.punctuation)
        return text.translate(translator)
    
    def remove_special_characters(self, text):
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text
    
    def remove_links(self, text):
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        return text
    
    def remove_html_tags(self, text):
        text = re.sub(r"<.*?>", "", text)
        return text
    
    def remove_extra_whitespaces(self, text):
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def remove_numbers(self, text):
        text = re.sub(r"\d+", "", text)
        return text
    def lemmatize_text(self, text):
        lemmatizer = WordNetLemmatizer()    
        lemmas = [lemmatizer.lemmatize(t) for t in text.split(" ")]
        lemmas = " ".join(lemmas)
        # make sure lemmas does not contains sotpwords
        text = self.remove_stopwords(lemmas)
        return text
    def preprocess_text(self, text):
        if self.rmStopWords:
          text = self.remove_stopwords(text)
        if self.lowercase:
          text = self.make_lowercase(text)
        if self.rmPuncs:
          text = self.remove_punctuation(text)
        if self.rmSpecialChars:
          text = self.remove_special_characters(text)
        if self.rmLinks:
          text = self.remove_links(text)
        if self.rmHtmlTags:
          text = self.remove_html_tags(text)
        text = self.space_bt_punct(text)
        if self.rmExtraSpaces:
          text = self.remove_extra_whitespaces(text)
        if self.rmNumbers:
          text = self.remove_numbers(text)
        if self.lemmatize:
          text = self.lemmatize_text(text)
        return text
    
def load_data(data_path):
        # Load the IMDb dataset from the data_path
        # The dataset is stored in a text file where each line contains a review and its label
        logging.info("Loading dataset")
        df = pd.read_csv(data_path, encoding='utf-8',delimiter=",")
        text_preprocesser = TextPreprocessor(rmStopWords = True, lowercase = True, rmPuncs = False,
                                                rmSpecialChars = False, rmLinks = True, rmHtmlTags = True,
                                                rmExtraSpaces = True, rmNumbers = False, lemmatize = True)
        logging.info("Preprocessing Text...")
        sentences = df.review.progress_apply(lambda x: text_preprocesser.preprocess_text(x)).tolist()
        labels = df.sentiment.apply(lambda x: 1 if x == "positive" else 0).tolist()
        return sentences, labels


class IMDBDataset(Dataset):
    def __init__(self, data, max_length=250):
        """
        Input: data: [[sentences],[labels]]
        """
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sentences, self.labels = data
        
    
    def preprocess(self, sentence):
        # Tokenize the sentence and truncate/pad it to the maximum length
        tokens = self.tokenizer.tokenize(sentence)[:self.max_length-2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        padding_length = self.max_length - len(token_ids)
        
        attention_mask = [1] * len(token_ids) + [0] * padding_length
        token_ids += [0] * padding_length
        
        return token_ids, attention_mask
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        sentence = self.sentences[index]
        label = self.labels[index]
        
        token_ids, attention_mask = self.preprocess(sentence)
        
        return {
            'input_ids': torch.tensor(token_ids),
            'attention_mask': torch.tensor(attention_mask),
            'label': torch.tensor(label)
        }
    @staticmethod
    def collate(batch):
      ids = []
      attention_masks=[]
      labels = []
      for item in batch:
        ids.append(item['input_ids'])
        attention_masks.append(item['attention_mask'])
        labels.append(item['label'])
      return {'ids': ids.to(device), 'attention_masks':attention_masks.to(device),'labels': labels.to(device)}
