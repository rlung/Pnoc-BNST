#%%

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from datetime import datetime
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

sig_fps = 5

# Time in open/close arm counted as transition
arm0_t_min = 5
center_t_limit = 5
arm1_t_min = 5

# Working directory
os.chdir(os.path.join(base_dir, 'PNOC'))

# Directory to save data
now = datetime.now()
save_dir = 'test/data-' + now.strftime('%y%m%d-%H%M%S')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)


#%% Import data from all animals
# Relies on all data acquisition to be the same between animals, eg frame #.

# Import behavior
behav_files = glob.glob('PNOC_EPM_Behavior/*.xlsx')
file_names = np.array([os.path.split(behav_file)[-1] for behav_file in behav_files])
file_num = len(file_names)

etho_out = map(etho_extract, behav_files)
behav_id_ord = [re.split('_|\.', file, 1)[0] for file in file_names]


#%% Format data

data = [etho[0] for etho in etho_out]
labels = [label.translate(None, '()<>/-').replace(' ', '_') for label in etho_out[0][1]]  # Assumes all labels are the same
time_ix = labels.index('Recording_time')
behav_frame_num = data[0].shape[0]
behav_fps = int(1 / np.diff(np.squeeze(data[0][:, time_ix])).mean())

# Normalize behavioral data
# Set range of values from -1 to 1. Exclude timestamps (first 2 columns).
data_norm = list(data)
for x in np.arange(file_num):
    var_maxes = np.nanmax(data_norm[x][:, 2:], axis=0, keepdims=True)
    data_norm[x][:, 2:] = data_norm[x][:, 2:] / np.repeat(var_maxes, behav_frame_num, axis=0)

# Import signals
sig_dirs = glob.glob('PNOC Traces_EPM/PNOC*')
sig_id_ord = [re.split('_|\.', os.path.split(sig_dir)[-1], 1)[0] for sig_dir in sig_dirs]
sigs_import = [np.load(os.path.join(sig_dir, 'extractedsignals.npy')) for sig_dir in sig_dirs]
sig_frame_num = np.min([x.shape[1] for x in sigs_import])

# Truncate sigs so all have the same number of frames
sigs_list = [sig[:, :sig_frame_num] for sig in sigs_import]

# Number of neurons per subject
num_cells_per_subj = [x.shape[0] for x in sigs_list]
cell_subj_id = [np.array(num_cells * [sig_id]) for num_cells, sig_id in zip(num_cells_per_subj, sig_id_ord)]

# Reshape so all neurons in one matrix
sigs = np.concatenate(tuple(sigs_list), axis=0)
sigs_id = np.concatenate(tuple(cell_subj_id), axis=0)
sig_ts = np.arange(sig_frame_num, dtype=float) / sig_fps

# Normalize (z score)
sigs_z = map(stats.zscore, sigs)
sigs_z = np.column_stack(sigs_z).T  # WATCH OUT FOR THE TRANSPOSE!!!


#%% Create new behavioral data

data_norm2 = list(data_norm)  # copy list

for d in np.arange(len(data_norm2)):
    # Make it work for list of data...
    open_col = labels.index('In_zoneOpen_arms__centerpoint')
    close_col = labels.index('In_zoneClosed_arms__centerpoint')
    
    # Identify state transitions
    # Index corresponds to the first behavioral frame of transition.
    open_diff = np.diff(np.squeeze(data_norm2[d][:, open_col]))
    close_diff = np.diff(np.squeeze(data_norm2[d][:, close_col]))
    
    # Behavioral frame of change 
    close_enter_frames = np.where(close_diff == 1)[0] + 1
    close_exit_frames = np.where(close_diff == -1)[0] + 1
    open_enter_frames = np.where(open_diff == 1)[0] + 1
    open_exit_frames = np.where(open_diff == -1)[0] + 1
    
    close_exit_num = len(close_exit_frames)
    open_enter_num = len(open_enter_frames)
    
    # Find close-to-open transitions
    close2open = np.zeros(behav_frame_num)
    last_frame = 0
    for frame in close_exit_frames:
        # pdb.set_trace()
    
        # Frame of close-arm entry
        # If no frame exists, must have started in closed arm, and first frame
        # should be used.
        close_enter_frames_before = np.where(frame - close_enter_frames > 0, close_enter_frames, 0)
        if np.any(close_enter_frames_before):
            close_enter_frame = np.max(close_enter_frames_before)
        else:
            close_enter_frame = 0
        
        # Frame of next closed-arm entry
        close_enter_frames_after = np.where(close_enter_frames - frame > 0,
                                            close_enter_frames, np.nan)
        if np.any(np.isfinite(close_enter_frames_after)):
            next_close_enter_frame = int(np.nanmin(close_enter_frames_after))
        else:
            next_close_enter_frame = behav_frame_num
        
        # Frame of next open-arm entry
        open_enter_frames_after = np.where(open_enter_frames - frame > 0,
                                           open_enter_frames, np.nan)              # first index in array of frame indices of open-arm entries after close-arm exit
        if np.any(np.isfinite(open_enter_frames_after)):
            next_open_enter_frame = int(np.nanmin(open_enter_frames_after))
        else:
            break
    
        # Frame of next open-arm exit
        open_exit_frames_after = np.where(open_exit_frames - frame > 0,
                                          open_exit_frames, np.nan)
        if np.any(np.isfinite(open_exit_frames_after)):
            next_open_exit_frame = int(np.nanmin(open_exit_frames_after))
        else:
            next_open_exit_frame = behav_frame_num
    
        # Check    
    #    print "\n{0}, {1}, {2}, {3}, {4}".format(close_enter_frame, frame, next_close_enter_frame, next_open_enter_frame, next_open_exit_frame)
    #    print [next_open_enter_frame < next_close_enter_frame,
    #           frame - close_enter_frame > arm0_t_min * behav_fps,
    #           next_open_enter_frame - frame < center_t_limit * behav_fps,
    #           next_open_exit_frame - next_open_enter_frame > arm1_t_min * behav_fps]
    
        # Must meet certain criteria to be classified as transition:
        # 1. Have been in closed arm for certain period of time (arm0_t_min)
        # 2. Must traverse center zone quickly (center_t_limit)
        # 3. Be in open arm for certain period of tie (arm1_t_min)
        if next_open_enter_frame < next_close_enter_frame and\
           frame - close_enter_frame > arm0_t_min * behav_fps and\
           next_open_enter_frame - frame < center_t_limit * behav_fps and\
           next_open_exit_frame - next_open_enter_frame > arm1_t_min * behav_fps:
            # print frame
            frame_start = frame - arm0_t_min * behav_fps
            frame_end = next_open_enter_frame + arm1_t_min * behav_fps
            close2open[frame_start:frame_end] = 1
            
        last_frame = frame
        # pdb.set_trace()
    
    # Find open-to-close transitions
    open2close = np.zeros(behav_frame_num)
    last_frame = 0
    for frame in open_exit_frames:
        # pdb.set_trace()
    
        # Frame of open-arm entry
        # If no frame exists, must have started in open arm, and first frame
        # should be used.
        open_enter_frames_before = np.where(frame - open_enter_frames > 0, open_enter_frames, 0)
        if np.any(open_enter_frames_before):
            open_enter_frame = np.max(open_enter_frames_before)
        else:
            open_enter_frame = 0
        
        # Frame of next open-arm entry
        open_enter_frames_after = np.where(open_enter_frames - frame > 0,
                                           open_enter_frames, np.nan)
        if np.any(np.isfinite(open_enter_frames_after)):
            next_open_enter_frame = int(np.nanmin(open_enter_frames_after))
        else:
            next_open_enter_frame = behav_frame_num
        
        # Frame of next closed-arm entry
        close_enter_frames_after = np.where(close_enter_frames - frame > 0,
                                            close_enter_frames, np.nan)              # first index in array of frame indices of open-arm entries after close-arm exit
        if np.any(np.isfinite(close_enter_frames_after)):
            next_close_enter_frame = int(np.nanmin(close_enter_frames_after))
        else:
            break
    
        # Frame of next closed-arm exit
        close_exit_frames_after = np.where(close_exit_frames - frame > 0,
                                           close_exit_frames, np.nan)
        if np.any(np.isfinite(close_exit_frames_after)):
            next_close_exit_frame = int(np.nanmin(close_exit_frames_after))
        else:
            next_close_exit_frame = behav_frame_num
    
        # Check    
    #    print "\n{0}, {1}, {2}, {3}, {4}".format(close_enter_frame, frame, next_close_enter_frame, next_open_enter_frame, next_open_exit_frame)
    #    print [next_open_enter_frame < next_close_enter_frame,
    #           frame - close_enter_frame > arm0_t_min * behav_fps,
    #           next_open_enter_frame - frame < center_t_limit * behav_fps,
    #           next_open_exit_frame - next_open_enter_frame > arm1_t_min * behav_fps]
    
        # Must meet certain criteria to be classified as transition:
        # 1. Have been in closed arm for certain period of time (arm0_t_min)
        # 2. Must traverse center zone quickly (center_t_limit)
        # 3. Be in open arm for certain period of tie (arm1_t_min)
        if next_close_enter_frame < next_open_enter_frame and\
           frame - open_enter_frame > arm0_t_min * behav_fps and\
           next_close_enter_frame - frame < center_t_limit * behav_fps and\
           next_close_exit_frame - next_close_enter_frame > arm1_t_min * behav_fps:
            # print frame
            frame_start = frame - arm0_t_min * behav_fps
            frame_end = next_close_enter_frame + arm1_t_min * behav_fps
            open2close[frame_start:frame_end] = 1
            
        last_frame = frame
    
    # Add new data
    data_norm2[d] = np.column_stack((data_norm2[d], close2open, open2close))

# Add labels of new variables
labels2 = labels + ['close2open_transition', 'open2close_transition']


#%% Downsample to match calcium imaging time

data_ds = [np.nan * np.zeros((sig_frame_num, len(labels2)), dtype=float)] * file_num

for d in np.arange(len(data_ds)):
    # Find matching "bin" in calcium imaging time for each time point of behavioral data
    bin_ix = np.digitize(data_norm2[d][:, time_ix], sig_ts)
    
    # Downsample
    for bin in np.arange(sig_frame_num):
        bin_pts = np.where(bin_ix == bin+1)[0]
        if bin_pts.size:
            data_ds[d][bin, :] = np.nanmean(data_norm2[d][bin_pts, :], axis=0)
    

#%% Clean and select data

selected_behavs = ['Velocity',
                   'In_zoneCenter__centerpoint',
                   'In_zoneOpen_arms__centerpoint',
                   'In_zoneClosed_arms__centerpoint',
                   'Mobility_stateHighly_mobile',
                   'Mobility_stateMobile',
                   'Mobility_stateImmobile',
                   'Distance_to_zone',
                   'close2open_transition',
                   'open2close_transition']
other_vars = ['Distance_to_zone * In_zoneOpen_arms__centerpoint',
              'Distance_to_zone * In_zoneClosed_arms__centerpoint']
num_vars = len(selected_behavs) + len(other_vars) + 3
formula = [[]] * file_num
valid_frames = [[]] * file_num

for d in np.arange(len(data_ds)):
    # Remove time points with missing data
    valid_frames[d] = np.where(np.isfinite(data_ds[d][:, time_ix]))[0]
    
    # Select behavioral variables
#    valid_behav_ix = np.ones(len(selected_behavs.shape, dtype=bool))
#    print "Behaviral variables chosen are:"
#    for n, behav in enumerate(selected_behavs):
#        if np.any(data_ds[d][valid_frames[d], labels2.index(behav)]):
#            print " + " + behav
#        else:
#            print " - " + behav + ": chosen but does not vary, thus omitted"
#            valid_behav_ix[n] = False
    
    # Define formula for subject
#    formula[d] = 'signal ~ const' + ' + '.join(selected_behavs[valid_behav_ix])\
    formula[d] = 'signal ~ signal_tminus1 + signal_tminus2 + ' +\
                 ' + '.join(selected_behavs + other_vars)


#%% Model

plot_dir = os.path.join(save_dir, 'plots')
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

num_cells = sum(num_cells_per_subj)

p_vals = np.zeros((num_cells, num_vars))
coeffs = np.zeros((num_cells, num_vars))

for cell in np.arange(num_cells):
    sig_subj_id = sigs_id[cell]
    subj_ix = behav_id_ord.index(sig_subj_id)
    
    # Behavioral data
    df = pd.DataFrame(data=data_ds[subj_ix][valid_frames[subj_ix], :], columns=labels2)
#    pdb.set_trace()
#    df = sm.add_constant(df)

    # Setup variables    
    y = sigs_z[cell, valid_frames[subj_ix]]
    df['signal'] = y
    df['signal_tminus1'] = np.concatenate((np.nan * np.ones(1), y[:-1]))
    df['signal_tminus2'] = np.concatenate((np.nan * np.ones(2), y[:-2]))

    # Signal lagging by frame
#    df['signal'] = np.concatenate((np.nan * np.ones(1), y[:-1]))
#    df['signal_tminus1'] = np.concatenate((np.nan * np.ones(2), y[:-2]))
#    df['signal_tminus2'] = np.concatenate((np.nan * np.ones(3), y[:-3]))
    
    # Run OLS
    model = smf.ols(data=df, formula=formula[subj_ix]).fit()  # first two frames has nan for signal_tminus2
    p_vals[cell, :] = model.pvalues
    coeffs[cell, :] = model.params
    
    # Plot
    prstd, iv_l, iv_u = wls_prediction_std(model)

    fig, ax  = plt.subplots(2, 1)
    fig.suptitle("Cell " + str(cell))
    
    ax[0].plot(df['signal'], 'b-', label="data")
    ax[0].plot(model.fittedvalues, 'r-')
    ax[0].plot(iv_l, 'r--')
    ax[0].plot(iv_u, 'r--')
    ax[0].set_title("Model")
    ax[0].set_ylabel("Fluorescence")
    
    ax[1].plot(df['signal'] - model.fittedvalues, 'g-')
    ax[1].set_title("Residual")
    ax[1].set_ylabel("Fluoresence residual")
    ax[1].set_xlabel("Time")

    plt.savefig(os.path.join(plot_dir, str(cell) + '.png'),
                dpi=200, bbox_inches='tight')

np.savetxt(os.path.join(save_dir, 'p_values.txt'), p_vals)
np.savetxt(os.path.join(save_dir, 'coeffs.txt'), coeffs)
np.savetxt(os.path.join(save_dir, 'formula.txt'), formula, fmt='%s')
