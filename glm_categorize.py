import numpy as np
import os

# Import custom functions
import sys
dirs2try = ['C:/Users/randa/Dropbox (Stuber Lab)/Documents/Python/',
            'D:/Dropbox (Stuber Lab)/Documents/Python/',
            '/data/Dropbox (Stuber Lab)/Documents/Python/']
base_dir = []
for dir2try in dirs2try:
    if os.path.isdir(dir2try):
        if dir2try not in sys.path:
            sys.path.append(dir2try)
        base_dir = dir2try
        break
if base_dir:
    from my_functions import bh_correction
else:
    print "Failed to define base directory"


#%% Parameters

p_thresh = 0.05


#%% Change directory

os.chdir(os.path.join(base_dir, 'PNOC'))


#%% Import data from file
# Data is formated as cells x variables

work_dir = 'test/data-160821-160633'
os.chdir(work_dir)
coeffs = np.loadtxt('coeffs.txt')
p_vals = np.loadtxt('p_values.txt')
var_names = np.genfromtxt('var_names.txt', delimiter='\t', dtype=str)


#%% Identify neurons significantly affected by each variable

sig_ix = bh_correction(p_vals.flatten())
sig_i, sig_j = np.unravel_index(sig_ix, p_vals.shape)

# Mask of significant variables for each neuron
sig_mask = np.zeros(p_vals.shape, dtype=bool)
sig_mask[sig_i, sig_j] = True

# Mask of positively and negatively significant variables
pos_sig_mask = np.logical_and(sig_mask, coeffs > 0)
neg_sig_mask = np.logical_and(sig_mask, coeffs < 0)

# Mask of significant varialbes with +1 and -1 for positve and negative
# relationship
dir_mask = np.zeros(coeffs.shape, dtype=int)
dir_mask[pos_sig_mask] = 1
dir_mask[neg_sig_mask] = -1


#%% 