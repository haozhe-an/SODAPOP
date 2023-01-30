# coding=utf-8
import jsonlines
from transformers import BertTokenizer
from transformers import BertForMultipleChoice
from transformers import RobertaTokenizer, RobertaForMaskedLM
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
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
import argparse

from utils import IQADataset, read_data, flat_accuracy, format_time
from names import weat_black, weat_white, weat_male, weat_female, hispanic, asian

arg_to_name = {'weat_female': weat_female,
               'weat_male' : weat_male,
               'weat_black': weat_black,
               'weat_white': weat_white,
               'asian': asian,
               'hispanic': hispanic}


def load_unmasker():
  model = RobertaForMaskedLM.from_pretrained("roberta-base")
  tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
  model = model.to(device)

  return tokenizer, model

def run_unmasker(unmasker_model, unmasker_tokenizer, sent_input, top_k):
  sent_input = sent_input.replace("[MASK]", "<mask>") 
  ret = []

  inputs = unmasker_tokenizer(sent_input, return_tensors="pt")
  inputs = inputs.to(device)
  with torch.no_grad():
    outputs = unmasker_model(**inputs)
  
  logits = outputs.logits
  mask_token_index = (inputs.input_ids == unmasker_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
  probs = logits[0, mask_token_index].softmax(dim=-1)
  values, predictions = probs.topk(top_k)

  decoded = unmasker_tokenizer.decode(predictions[0]).split(" ")

  for v, p, d in zip(values[0], predictions[0], decoded):
    result = {}
    result["score"] = float(v)
    result["token"] = int(p)
    result["token_str"] = d
    sent_input = sent_input.replace("<mask>", "[MASK]") 
    result["sequence"] = sent_input.replace("[MASK]", d)
    ret.append(result)
  return ret

def find_neighbors(sent, k, context=None, question=None): 
  if k == 0:
    return [sent]
  if k >= 2:
    base = find_neighbors(sent, 1, context=context, question=question)
    neighbors = deepcopy(base)
    for neighbor in base:
      neighbors.extend(find_neighbors(neighbor, k-1, context=context, question=question))
    return neighbors

  tokens = word_tokenize(sent)
  neighbors = []
  for i in range(len(tokens)):
    input_sent = ' '.join(tokens[:i] + ['[MASK]'] + tokens[(i+1):])
    if context is not None:
      input_sent = context + ' ' + question + ' ' + input_sent
    #print(input_sent)
    pred_sent = run_unmasker(unmasker_model, unmasker_tokenizer, input_sent, 10)
    #print(pred_sent)
    #assert len(pred_sent) == 8
    neighbors.extend([' '.join(word_tokenize(p["sequence"])[-len(tokens):]) for p in pred_sent])
  return neighbors


def experiment(dest_file, tokenizer):
  cwd = os.getcwd()    
  prob_file_name = "{c}_prob".format(c=dest_file.strip('.jsonl').split('/')[-1])
  label_file_name = "{c}_label".format(c=dest_file.strip('.jsonl').split('/')[-1])
  prob_file = os.path.join(cwd, args.pred_dir, prob_file_name)
  label_file = os.path.join(cwd, args.pred_dir, label_file_name)
  input_file = dest_file

  dev_encodings, dev_labels = read_data(input_file, tokenizer)
  dev_dataset = IQADataset(dev_encodings, dev_labels)
  from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
  batch_size = 8
  dev_dataloader = DataLoader(
              dev_dataset, # The validation samples.
              sampler = SequentialSampler(dev_dataset), # Pull out batches sequentially.
              batch_size = batch_size # Evaluate with this batch size.
          )

  print("")
  print("Running Validation...")
  t0 = time.time()
  model.eval()
 
  total_eval_accuracy = 0
  total_eval_loss = 0
  nb_eval_steps = 0

  probability_output = []
  labels_output = []

  for batch in dev_dataloader:
      with torch.no_grad():        
          encodings, labels = batch
          outputs = model(encodings['input_ids'].squeeze(1).to(device), 
                          token_type_ids=encodings['token_type_ids'].squeeze(1).to(device),
                          attention_mask=encodings['attention_mask'].squeeze(1).to(device),
                          labels=labels.to(device)
                          )
      loss = outputs.loss
      logits = outputs.logits    
      total_eval_loss += loss.item()

      probability = m(outputs.logits)
      probability = probability.detach().cpu().numpy()
      probability_output.extend(probability)
      labels_output.extend(labels)
      logits = logits.detach().cpu().numpy()
      label_ids = labels.to('cpu').numpy()
      total_eval_accuracy += flat_accuracy(logits, label_ids)
      
  print("Input file is {input_file} \nTotal number samples being evaluated: {num}".format(input_file=input_file, num=len(dev_encodings)))
  avg_val_accuracy = total_eval_accuracy / len(dev_dataloader)
  print("  Accuracy: {0:.4f}".format(avg_val_accuracy))

  avg_val_loss = total_eval_loss / len(dev_dataloader)
  validation_time = format_time(time.time() - t0)

  print("  Validation Loss: {0:.2f}".format(avg_val_loss))
  print("  Validation took: {:}".format(validation_time))
  print("")
  print("")
  print("")

  np.save(prob_file, np.array(probability_output))
  np.save(label_file, np.array(labels_output))
  return avg_val_accuracy, avg_val_loss

def replace(datum, orig_name, new_name):
  for key in datum.keys():
    datum[key] = datum[key].replace(orig_name, new_name)
  return datum


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_file', required=True, type=str, default=None,
                        help="original dev set to be used for distractor generation")
    parser.add_argument('--model_checkpoint', required=True, type=str, default=None,
                        help="MCQ model checkpoint")
    parser.add_argument('--name_group', required=True, type=str, default=None,
                        choices=['weat_female', 'weat_male', 'weat_black', 'weat_white',
                                'asian', 'hispanic'],
                        help="seed names to be used")
    parser.add_argument('--output_dir', required=True, type=str, default=None,
                        help="output directory where new MCQs with generated distractors are saved; use relative path")
    parser.add_argument('--pred_dir', required=True, type=str, default=None,
                        help="output directory where predictions are saved; use relative path")
    
    args = parser.parse_args()
    
    cwd = os.getcwd()
    for path in [args.output_dir, args.pred_dir]:
        directory = os.path.join(cwd, path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    if torch.cuda.is_available():     
           device = torch.device("cuda")
           print('There are %d GPU(s) available.' % torch.cuda.device_count())
           print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
           print('No GPU available, using the CPU instead.')
           device = torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
    model.load_state_dict(torch.load(args.model_checkpoint))
    print("Loaded BERT tokenizer and model")
    model.cuda()
    m = torch.nn.Softmax(dim=1)

    unmasker_tokenizer, unmasker_model= load_unmasker()

    dev_file = args.dev_file
    seed_names = arg_to_name[args.name_group]

    for name in seed_names:
          with jsonlines.open(dev_file) as f:
              for i, datum in enumerate(f.iter()):
                
                  # use the first 50 MCQs that have the specific question below
                  if i > 320:
                      break
                  # only work on MCQs with this specific type of question
                  # Bethany is a random name
                  if not "How would you describe Bethany?" in datum["question"]:
                      continue 

                  random.seed(10)
                  distractor_file_name = 'new_MCQ_{}_ex{}.jsonl'.format(name, i)
                  prob_file_name = '{c}_prob.npy'.format(c=distractor_file_name.strip('.jsonl'))
                
                  dest_file = os.path.join(cwd, args.output_dir, distractor_file_name)
                  prob_file = os.path.join(cwd, args.pred_dir, prob_file_name)
                
                  print(dest_file, prob_file)

                  if os.path.exists(prob_file):
                     print("Skipping {} sample {}".format(name, i))
                     continue

                  if not os.path.exists(dest_file):
                     augment_data = []
                     correct_key = 'answer{}'.format(datum["correct"])
                     correct_ans = datum[correct_key]

                     datum = replace(datum, "Bethany", name)
                     candidates = find_neighbors(correct_ans, 3, context=datum["context"],question="{} is".format(name)) 
                     candidates = list(set(candidates))
                     random.shuffle(candidates)
                     candidates = candidates[:min(10000, len(candidates)//2*2)]
                        
                     # construct new MCQs using the distractors genearated
                     wrong_ans_key = set(["A", "B", "C"]) - set(datum["correct"])
                     wrong_ans_key = ['answer{}'.format(a) for a in wrong_ans_key]
                     for j in range(0, len(candidates), 2):
                       datum[wrong_ans_key[0]] = candidates[j]
                       datum[wrong_ans_key[1]] = candidates[j+1]
                       augment_data.append(deepcopy(datum))

                     with jsonlines.open(dest_file, mode='w') as writer:
                       writer.write_all(augment_data)

                  experiment(dest_file, tokenizer)