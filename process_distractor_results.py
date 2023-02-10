import jsonlines
from transformers import BertTokenizer
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import json
import glob
import random
from copy import deepcopy
from collections import defaultdict
from scipy.spatial import distance
import string
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from p_test import p_value_sample
from names import weat_white_female, weat_white_male, weat_black_female, weat_black_male
from names import hispanic_male, hispanic_female, asian_male, asian_female
from names import weat_black, weat_white, weat_male, weat_female, hispanic, asian

import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords') 

output_dir = "fig_output"
pred_model = "bert"


def load_from_dir(target_dir_list, pred_dir_list):
    person_to_wrong = defaultdict(list)
    person_to_all = defaultdict(list)
    
    for target_dir, pred_dir in zip(target_dir_list, pred_dir_list):
        check = pred_dir + '{}_person_to_all.json'.format(target_dir.strip("/").split("/")[-1])
        print("tar", target_dir, flush=True)
        print("pred", check, flush=True)
        
        if os.path.exists(check):
            print("found existing results in json format")
            with open(pred_dir + '{}_person_to_all.json'.format(target_dir.strip("/").split("/")[-1])) as f:
                temp = json.load(f)
                for k,v in temp.items():
                    person_to_all[k] = v
            with open(pred_dir + '{}_person_to_wrong.json'.format(target_dir.strip("/").split("/")[-1])) as f:
                temp = json.load(f)
                for k,v in temp.items():
                    person_to_wrong[k] = v
        else:
            temp_person_to_wrong = defaultdict(list)
            temp_person_to_all = defaultdict(list)
            for file in tqdm(glob.glob(target_dir + "*.jsonl")):
                if "person_to_" in file:
                    continue
                _, _, _, person, ex = file.split("/")[-1].strip(".jsonl").split("_")
                ex = ex.strip("ex")
                probs = np.load(pred_dir + "new_MCQ_{person}_ex{ex}_prob.npy".format(person=person, ex=ex))
                labels = np.load(pred_dir + "new_MCQ_{person}_ex{ex}_label.npy".format(person=person, ex=ex))
                assert len(probs) == len(labels)
                
                with jsonlines.open(file) as f:
                    i = 0
                    correct = 0
                    wrong_pred = set()
                    all_distractors = set()
                    for datum in f.iter():
                        for distractor_label in range(3):
                            if distractor_label == labels[i]: 
                                continue
                            all_distractors.add(datum[pred_to_answer[distractor_label]])
                        p = np.argmax(probs[i])
                        if p != labels[i]:
                            wrong_pred.add(datum[pred_to_answer[p]])
                        else:
                            correct += 1
                        i += 1
                    temp_person_to_wrong[person].extend(list(deepcopy(wrong_pred)))
                    temp_person_to_all[person].extend(list(deepcopy(all_distractors)))
            with open(pred_dir + '{}_person_to_all.json'.format(target_dir.strip("/").split("/")[-1]), "w") as f:
                json.dump(temp_person_to_all, f)
                print("wrote person_to_all to json file", flush=True)

            with open(pred_dir + '{}_person_to_wrong.json'.format(target_dir.strip("/").split("/")[-1]), "w") as f:
                json.dump(temp_person_to_wrong, f)
                print("wrote persont_to_wrong to json file", flush=True)                    
                                
            for k,v in temp_person_to_all.items():
                person_to_all[k] = v
            for k,v in temp_person_to_wrong.items():
                person_to_wrong[k] = v 
                
    print("len(person_to_all), len(person_to_wrong)", len(person_to_all), len(person_to_wrong), flush=True)            
    return person_to_all, person_to_wrong

def process_person_to_vocab(person_to_all, person_to_wrong):
    person_to_vocab_wrong = {}
    person_to_vocab_all = {}
    for person in person_to_wrong:
        person_to_vocab_wrong[person] = defaultdict(int)
        person_to_vocab_all[person] = defaultdict(int)
        for choice in person_to_wrong[person]:
            for word in choice.split(" "):
                person_to_vocab_wrong[person][word] += 1
        for choice in person_to_all[person]:
            for word in choice.split(" "):
                person_to_vocab_all[person][word] += 1

        for word in person_to_vocab_wrong[person]:
            person_to_vocab_wrong[person][word] /= person_to_vocab_all[person][word]

        print("{} len(person_to_vocab_all[person]), len(person_to_vocab_wrong[person])".format(person), 
              len(person_to_vocab_all[person]), len(person_to_vocab_wrong[person]))
    return person_to_vocab_all, person_to_vocab_wrong

def get_all_vocab(person_to_vocab_all, stop_words):
    all_vocab = []

    for w in list(person_to_vocab_all[list(person_to_vocab_all.keys())[0]].keys()):
        if w in stop_words or w in string.punctuation:
            continue
        should_continue = False
        for char in w:
            if char in string.punctuation and char != "-":
                should_continue = True
                break
        if should_continue:
            continue

        if (np.array([person_to_vocab_all[k][w] for k in person_to_vocab_all.keys()]) > 50).all():
            all_vocab.append(w)

    print("len(all_vocab)", len(all_vocab), flush=True)
    return all_vocab

if __name__ == "__main__":
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    pred_to_answer = {
            0 : "answerA",
            1 : "answerB",
            2 : "answerC"}
    stop_words = set(stopwords.words('english'))
    
    # target dir contains newly construct MCQ samples
    # pred dir contains predictions (logits and label)
    # write them as lists so that multiple settings could be analyzed simultaneously
    target_dir_list = ["demo_output"]
    
    pred_dir_list_all = [["demo_pred"]]
    settings_all = ["EA_AA", "EA_AS", "EA_HS", "HS_AA", "HS_AS", "AS_AA"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for pred_dir_list in [pred_dir_list_all[0]]: 
        
        person_to_all, person_to_wrong = load_from_dir(target_dir_list, pred_dir_list)
        person_to_vocab_all, person_to_vocab_wrong = process_person_to_vocab(person_to_all, person_to_wrong)
        all_vocab = get_all_vocab(person_to_vocab_all, stop_words)
        
        person_to_vec = {}
        for person in person_to_vocab_wrong:
            person_to_vec[person] = []
            for word in all_vocab:
                person_to_vec[person].append(person_to_vocab_wrong[person][word])
            assert len(person_to_vec[person]) == len(all_vocab)
            person_to_vec[person] = np.array(person_to_vec[person])
            
        AA_female = [person_to_vec[person] for person in weat_black_female]
        EA_female = [person_to_vec[person] for person in weat_white_female]
        AA_male = [person_to_vec[person] for person in weat_black_male]
        EA_male = [person_to_vec[person] for person in weat_white_male]
        AS_female = [person_to_vec[person] for person in asian_female[:25]]
        AS_male = [person_to_vec[person] for person in asian_male[:25]]
        HS_female = [person_to_vec[person] for person in hispanic_female[:25]]
        HS_male = [person_to_vec[person] for person in hispanic_male[:25]]
        
        # Kmeans clustering
        compare = [(EA_female, EA_male), (AA_female, AA_male),
                   (HS_female, HS_male), (AS_female, AS_male), 
                   (EA_female, AA_female), (EA_male, AA_male),
                   (EA_female, AA_male), (EA_male, AA_female),
                   (EA_female, AS_female), (EA_male, AS_male),
                   (EA_female, HS_female), (EA_male, HS_male),
                   (AA_female, AS_female), (AA_male, AS_male),
                   (AA_female, HS_female), (AA_male, HS_male),
                   (HS_female, AA_female), (HS_male, AA_male),
                   ]
        legend = ['(EA_female, EA_male)', '(AA_female, AA_male)',
                  '(HS_female, HS_male)', '(AS_female, AS_male)',
                   '(EA_female, AA_female)', '(EA_male, AA_male)',
                   '(EA_female, AA_male)', '(EA_male, AA_female)',
                   '(EA_female, AS_female)', '(EA_male, AS_male)',
                   '(EA_female, HS_female)', '(EA_male, HS_male)',
                   '(AA_female, AS_female)', '(AA_male, AS_male)',
                   '(AA_female, HS_female)', '(AA_male, HS_male)',
                   '(HS_female, AA_female)', '(HS_male, AA_male)',
                   ]
        for l, (group1, group2) in zip(legend, compare):
            X_two_grp = np.array(group1 + group2)
            y_true = [1] * len(group1) + [0] * len(group2)

            kmeans_1 = KMeans(n_clusters=2, random_state=0).fit(X_two_grp)
            y_pred_1 = kmeans_1.predict(X_two_grp)
            correct = [1 if item1 == item2 else 0 for (item1, item2) in zip(y_true, y_pred_1)]
            acc = max(sum(correct) / float(len(correct)), 1 - sum(correct) / float(len(correct)))
            print("{}\t{}\n".format(l, acc))

        for setting in settings_all:
            group1 = setting.split("_")[0]
            group2 = setting.split("_")[1]

            print(setting, group1, group2, flush=True)

            # tSNE visualization
            if setting == "EA_AA":
                X = np.array(EA_female + AA_female + EA_male + AA_male)
                token_len = np.array([len(tokenizer.tokenize(person)) for person in weat_white_female 
                                      + weat_black_female + weat_white_male + weat_black_male]) 
            elif setting == "EA_AS":
                X = np.array(EA_female + AS_female + EA_male + AS_male)
                token_len = np.array([len(tokenizer.tokenize(person)) for person in weat_white_female 
                                      + asian_female + weat_white_male + asian_male]) 
            elif setting == "EA_HS":
                X = np.array(EA_female + HS_female + EA_male + HS_male)
                token_len = np.array([len(tokenizer.tokenize(person)) for person in weat_white_female 
                                      + hispanic_female + weat_white_male + hispanic_male]) 
            elif setting == "AS_AA":
                X = np.array(AS_female + AA_female + AS_male + AA_male)
                token_len = np.array([len(tokenizer.tokenize(person)) for person in asian_female 
                                      + weat_black_female + asian_male + weat_black_male])
            elif setting == "HS_AA":
                X = np.array(HS_female + AA_female + HS_male + AA_male)
                token_len = np.array([len(tokenizer.tokenize(person)) for person in hispanic_female 
                                      + weat_black_female + hispanic_male + weat_black_male])
            elif setting == "HS_AS":
                X = np.array(HS_female + AS_female + HS_male + AS_male)
                token_len = np.array([len(tokenizer.tokenize(person)) for person in hispanic_female 
                                      + asian_female + hispanic_male + asian_male])

            X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', random_state=10).fit_transform(X)

            plot_X = [X_embedded[0:25], X_embedded[25:50], X_embedded[50:75], X_embedded[75:100]]

            figure(figsize=(8, 6), dpi=80)
            msize = 80
            scatter = plt.scatter(plot_X[0][:,0], plot_X[0][:,1], c='tab:blue', 
                                  marker="x", label="{} Female".format(group1), s=msize)
            scatter = plt.scatter(plot_X[1][:,0], plot_X[1][:,1], c='tab:orange', 
                                  marker="x", label="{} Female".format(group2), s=msize)
            scatter = plt.scatter(plot_X[2][:,0], plot_X[2][:,1], c='tab:cyan', 
                                  marker="o", label="{} Male".format(group1), s=msize)
            scatter = plt.scatter(plot_X[3][:,0], plot_X[3][:,1], c='tab:red', 
                                  marker="o", label="{} Male".format(group2), s=msize)
            plt.legend(fontsize=20)
            plt.tight_layout()
            
            out = "./{}/srv_gen_roberta_pred_{}_{}_{}.pdf".format(output_dir, pred_model, group1, group2)
            if os.path.exists(out):
                print("Skipping because file {} already exists...".format(out))
            else:
                plt.savefig(out) 
            plt.clf()

            figure(figsize=(10, 6), dpi=80)
            cmap = plt.get_cmap('tab20', np.max(token_len) - np.min(token_len) + 1)
            scatter = plt.scatter(np.concatenate([plot_X[0][:,0], plot_X[2][:,0]]), 
                                  np.concatenate([plot_X[0][:,1], plot_X[2][:,1]]), 
                                  c=np.concatenate([token_len[0:25], token_len[50:75]]), marker="^", label=group1, s=msize,
                                  cmap=cmap,
                                  vmin=np.min(token_len) - 0.5, 
                                  vmax=np.max(token_len) + 0.5)
            scatter = plt.scatter(np.concatenate([plot_X[1][:,0], plot_X[3][:,0]]), 
                                  np.concatenate([plot_X[1][:,1], plot_X[3][:,1]]), 
                                  c=np.concatenate([token_len[25:50], token_len[75:100]]), marker="*", label=group2, s=msize,
                                  cmap=cmap,
                                  vmin=np.min(token_len) - 0.5, 
                                  vmax=np.max(token_len) + 0.5)
            plt.legend(fontsize=24)
            cb = plt.colorbar(scatter, ticks=np.arange(np.min(token_len), np.max(token_len) + 1))
            cb.set_label(label="Tokenization length", fontsize=26)
            cb.ax.tick_params(labelsize=22)
            plt.tight_layout()
            out = "./{}/srv_gen_roberta_pred_{}_{}_{}.pdf".format(output_dir, pred_model, group1, group2)
            if os.path.exists(out):
                print("Skipping because file {} already exists...".format(out))
            else:
                plt.savefig(out) 
            plt.clf()

            

        # permutation test
        str_to_vec = {"EA_female": EA_female, "EA_male": EA_male,
                      "AA_female": AA_female, "AA_male": AA_male,
                      "AS_female": AS_female, "AS_male": AS_male,
                      "HS_female": HS_female, "HS_male": HS_male
                     }
        str_to_name = {"EA_female": weat_white_female, "EA_male": weat_white_male,
                       "AA_female": weat_black_female, "AA_male": weat_black_male,
                       "AS_female": asian_female[:25], "AS_male":  asian_male[:25],
                       "HS_female": hispanic_female[:25], "HS_male": hispanic_male[:25]
                      }
        
        group_pairs = [("AA_female", "EA_female"), ("AS_female", "EA_female"), ("HS_female", "EA_female"),
                       ("AA_male", "EA_male"), ("AS_male", "EA_male"), ("HS_male", "EA_male"),
                       ("AA_female", "AS_female"), ("AA_female", "HS_female"), ("HS_female", "AS_female"),
                       ("AA_male", "AS_male"), ("AA_male", "HS_male"), ("HS_male", "AS_male"),
                       ("EA_female", "EA_male"), ("AA_female", "AA_male"), ("AS_female", "AS_male"), ("HS_female", "AS_male"),
                      ]
        for group in group_pairs:
            group1 = str_to_vec[group[0]]
            group2 = str_to_vec[group[1]]
            group1_names = str_to_name[group[0]]
            group2_names = str_to_name[group[1]]
            
            print(group)
            print(group1_names)
            print(group2_names, flush=True)                
            
            vocab_to_relative_diff = {}
            for i, (diff, v) in enumerate(zip(np.sum(np.array(group1) - np.array(group2), axis=0), all_vocab)):
                if np.sum(group1, axis=0)[i] == 0 or np.sum(group2, axis=0)[i] == 0:
                    continue
                if diff != 0:
                    vocab_to_relative_diff[v] = diff/np.average( [(np.sum(group1, axis=0)[i])/len(group1), (np.sum(group2, axis=0)[i])/len(group2)] )
            sorted_vocab_to_relative_diff = dict(sorted(vocab_to_relative_diff.items(), key=lambda item: item[1]))
            
            # by default, it is sorted in ascending order
            print("Words associated with group2 i.e. {}.".format(group[1]))
            for word, ratio in list(sorted_vocab_to_relative_diff.items())[:15]:
                print("{}\t{:.3f}".format(word, ratio))
            to_print = list(sorted_vocab_to_relative_diff.items())[-15:]
            to_print.reverse()
            print("Words associated with group1 i.e. {}.".format(group[0]))
            for word, ratio in to_print:
                print("{}\t{:.3f}".format(word, ratio))
                
            print("permutation test in progress...", flush=True)
            print("Words associated with group2 i.e. {}.".format(group[1]))
            for word, ratio in list(sorted_vocab_to_relative_diff.items())[:15]:
                p_value = p_value_sample(group1_names, group2_names, person_to_vocab_wrong, word)
                print("{}\t{:.3f}\t{:.8f}\n".format(word, ratio, p_value))
                
                
            to_print = list(sorted_vocab_to_relative_diff.items())[-15:]
            to_print.reverse()
            print("Words associated with group1 i.e. {}.".format(group[0]))
            for word, ratio in to_print:
                p_value = p_value_sample(group1_names, group2_names, person_to_vocab_wrong, word)
                print("{}\t{:.3f}\t{:.8f}\n".format(word, ratio, p_value))