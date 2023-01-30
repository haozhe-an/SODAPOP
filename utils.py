import jsonlines
import os
import pandas as pd
import numpy as np
import json
import glob
import nltk
nltk.download('punkt')
from copy import deepcopy
from nltk.tokenize import word_tokenize
import random
import time
import datetime
import torch


label_converter = {'A': torch.tensor(0).unsqueeze(0),
                   'B': torch.tensor(1).unsqueeze(0),
                   'C': torch.tensor(2).unsqueeze(0)}

class IQADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = self.encodings[idx]
        return item, self.labels[idx]

    def __len__(self):
        return len(self.labels)
    
def read_data(file_name, tokenizer):
  labels = []
  encodings = []
  with jsonlines.open(file_name) as f:
      for idx, datum in enumerate(f.iter()):
        label = label_converter[datum['correct']]
        encoding = tokenizer([datum['context'] + datum['question'],
                              datum['context'] + datum['question'], 
                              datum['context'] + datum['question']], 
                            [datum['answerA'], datum['answerB'], datum['answerC']], 
                            return_tensors='pt', 
                            max_length = 128,
                            pad_to_max_length = True,
                            )
        labels.append(label)
        encodings.append(encoding)
      labels = torch.tensor(labels)
  return encodings, labels

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

