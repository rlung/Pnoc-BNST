#%%

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime
# from openpyxl import load_workbook
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import pdb

# Import custom functions
import sys
dirs2try = ['C:/Users/randa/Dropbox (Stuber Lab)/Documents/Python/',
            'D:/Dropbox (Stuber Lab)/Documents/Python/',
            '/data/Dropbox (Stuber Lab)/Documents/Python/']
base_dir = []
for dir2try in dirs2try:
    if os.path.isdir(dir2try):
        sys.path.append(dir2try)
        base_dir = dir2try
        break
if base_dir:
    from my_functions import etho_extract
else:
    print "my_function not imported. Directory not identified."


#%% Parameters

# Working directory
os.chdir(os.path.join(base_dir, 'PNOC'))

# Directory to save data
now = datetime.now()
save_dir = 'data/data-' + now.strftime('%y%m%d-%H%M%S')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Import behavior
behav_files = glob.glob('PNOC_Odor_Behavior/PNOC187*.xlsx')
etho_out = map(etho_extract, behav_files)

data = np.concatenate(tuple([file[0] for file in etho_out]), axis=0)
labels = etho_out[0][1]  # Assumes all labels are the same

time_ix = np.where(np.array(labels) == 'Trial time')
ts_by_epoch = [np.squeeze(file[0][:, time_ix]) for file in etho_out] + np.array([0, 300, 600])
ts = np.concatenate(tuple(ts_by_epoch), axis=0)

# Import signals
sig_fps = 5
sig_dirs = 'PNOC Traces_Odor/PNOC185'
sigs = np.load(os.path.join(sig_dirs, 'extractedsignals.npy'))
num_time_pts = sigs.shape[1]
sig_ts = np.arange(num_time_pts, dtype=float) / sig_fps

# Normalize
sigs_z = map(stats.zscore, sigs)
sigs_z = np.column_stack(sigs_z).T

# Identify epochs in behavior data
epochs_temp = [np.ones(epoch.shape) * ii for ii, epoch in enumerate(ts_by_epoch)]
epochs = np.concatenate(tuple(epochs_temp), axis=0)

# Add data
base_ix = np.where(epochs == 0, 1, 0)
water_ix = np.where(epochs == 1, 1, 0)
tmt_ix = np.where(epochs == 2, 1, 0)
data = np.column_stack((data, base_ix, water_ix, tmt_ix))
labels = labels + ['Baseline epoch', 'Water epoch', 'TMT epoch']

#%% Create new data





#%% Downsample to match calcium imaging time

# Find matching "bin" in calcium imaging time for each time point of behavioral data
bin_ix = np.digitize(ts, sig_ts)

# Downsample
data_new = np.nan * np.zeros((num_time_pts, len(labels)), dtype=float)
for bin in np.arange(num_time_pts):
    bin_pts, = np.where(bin_ix == bin+1)
    if bin_pts.size:
        data_new[bin, :] = np.nanmean(data[bin_pts, :], axis=0)

# Add new data
#data_new = np.column_stack((data_new, sigs_minus1[0, :], sigs_minus2[0, :])
#labels = labels + ['signal_tminus1', 'signal_tminus2']
data_new = np.column_stack((data_new, sig_ts))
label_new = labels + ['Time']
    

#%% Clean and select data

# Remove time points with missing data
nan_ix = np.isnan(data_new[:, 2])
not_tmt_ix = np.logical_not(data_new[:, -2])  # index of non tmt epochs
del_ix = np.where(np.logical_or(nan_ix, not_tmt_ix))
data_clean = np.delete(data_new, del_ix, 0)
sig_clean = np.delete(sigs_z, del_ix, 1)

# Normalize data
data_norm = np.zeros(data_clean.shape)
for behav in range(data_clean.shape[1]):
    behav_data = data_clean[:, behav]
    data_min = np.nanmin(behav_data)
    behav_data_shift = behav_data - data_min
    data_max = np.nanmax(behav_data_shift)
    data_norm[:, behav] = behav_data_shift / data_max

# Select data
selected_ix = [15, 16, 17, 18, 27]
num_ts = data_clean.shape[0]
selected_data = data_clean[np.arange(num_ts).reshape(-1, 1), selected_ix]
selected_labels = [label_new[ix] for ix in selected_ix]

# Check if there are any variables with only 0's
uniques = map(np.unique, selected_data.T)
not_empties = map(np.any, uniques)
empties = np.logical_not(not_empties)  # Index of variables that are all zeros


print "Warning: The following variables were removed because they contain all zeros."
for ix in np.where(empties)[0]:
    print "  - " + selected_labels[ix]

selected_data = np.delete(selected_data, np.where(empties)[0], axis=1)
selected_labels = np.delete(selected_labels, np.where(empties)[0], axis=0)


#%% Model

df = pd.DataFrame(selected_data[2:, :], columns=selected_labels)
df.set_index('Time')

plot_dir = 'plots'
num_vars = df.shape[1] + 3
num_cells = sigs.shape[0]

X = df
X['constant'] = np.ones(num_ts-2)
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)
results = np.zeros((num_cells, num_vars, 3))
for cell in np.arange(num_cells):
    # Setup variables    
    y = sig_clean[cell, 2:]
    X['sig_tminus1'] = sig_clean[cell, 1:-1]
    X['sig_tminus2'] = sig_clean[cell, :-2]
    
    # Run OLS
    model = sm.OLS(y, X, missing='drop').fit()
    results[cell, ...] = np.column_stack((model.tvalues,
                                          model.pvalues,
                                          model.params))
    
    # Plot
    prstd, iv_l, iv_u = wls_prediction_std(model)
    nan_ix = np.where(np.isnan(df))[0]
    y_new = np.delete(y, nan_ix, axis=0)

    fig, ax  = plt.subplots(2, 1)
    ax[0].plot(y_new, 'b-', label="data")
    ax[0].plot(model.fittedvalues, 'r-')
    ax[0].plot(iv_l, 'r--')
    ax[0].plot(iv_u, 'r--')
    ax[1].plot(y_new-model.fittedvalues, 'g-')
    plt.savefig(os.path.join(save_dir, plot_dir, str(cell) + '.png'),
                dpi=200, bbox_inches='tight')

np.save(os.path.join(save_dir, 'model_results.txt'), results)
np.save(os.path.join(save_dir, 'model_vars.txt'), df.columns)
