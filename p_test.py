import random
import scipy
import numpy as np

def compute_ratio(X, Y, person_to_vocab_wrong, word):
  X_avg = np.average([person_to_vocab_wrong[x][word] for x in X])
  Y_avg = np.average([person_to_vocab_wrong[y][word] for y in Y])
  ratio = X_avg / Y_avg
  return ratio

def compute_diff(X, Y, person_to_vocab_wrong, word):
  X_avg = np.average([person_to_vocab_wrong[x][word] for x in X])
  Y_avg = np.average([person_to_vocab_wrong[y][word] for y in Y])
  diff = X_avg - Y_avg
  return diff

def p_value_sample(X, Y, person_to_vocab_wrong, word):
    
    random.seed(10)
    np.random.seed(10)
    assert(len(X) == len(Y))
    length = len(X) 
    
    diff_orig = compute_diff(X, Y, person_to_vocab_wrong, word)

    num_of_samples = min(1000000, int(scipy.special.comb(length*2,length)*100))

    larger = 0
    for i in range(num_of_samples):
        permute = np.random.permutation(X+Y)
        Xi = permute[:length]
        Yi = permute[length:]

        diff_new = compute_diff(Xi, Yi, person_to_vocab_wrong, word)
        if abs(diff_new) > abs(diff_orig):
          larger += 1
          
    return larger/float(num_of_samples) 
