#%%

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib_venn import venn3
import seaborn as sns
import glob
import os
import re
from copy import deepcopy
from functools import partial
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
    from my_functions import etho_extract, activity_map, interp_missing, bh_correction
else:
    print "my_function not imported. Directory not identified."

# Delete unnecessary variables
dir2try = dirs2try = None
del dir2try, dirs2try


#%% Parameters

### Define data ###
event_data = True
task = 'TMT'  # Choices: EPM, TMT, PNT, Water_TMT, Water_PNT

# Working directory
os.chdir(os.path.join(base_dir, 'PNOC'))
behav_files = glob.glob('PNOC_TMT/PNOC_TMT_Behavior/*tmt.xlsx')
sig_dirs = glob.glob('PNOC_TMT/PNOC_TMT_Events/PNOC*')

subj_cell_file = 'PNOC_TMT/animal_cell_numbers.txt'
tmt_exc_file = 'PNOC_TMT/tmt_exc_cells.txt'
tmt_inh_file = 'PNOC_TMT/tmt_inh_cells.txt'
###################

# Check number of files are equal
file_num_sig = len(sig_dirs)
file_num_behav = len(behav_files)
if file_num_sig != file_num_behav:
    print "Number of signal ({}) and behavioral ({}) files do NOT match!".format(file_num_sig, file_num_behav)

sig_fps = 5
time_label = 'Recording_time'

if task == 'EPM':
    sig_start = 0
elif task == 'TMT' or\
     task == 'PNT':
    sig_start = 3000
elif 'Water' in task:
    sig_start = 1500

# Time in open/close arm counted as transition
arm0_t_min = 5
center_t_limit = 5
arm1_t_min = 5

# Directory to save data
now = datetime.now()
save_dir = 'data/{}_glm-'.format(task) + now.strftime('%y%m%d-%H%M%S')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)


#%% Import data from all animals
# Relies on all data acquisition to be the same between animals, eg frame #.

# Import behavior
file_names = np.array([os.path.split(behav_file)[-1] for behav_file in behav_files])
file_num = len(file_names)

etho_out = map(etho_extract, behav_files)
behav_id_ord = [re.split('_|\.', file, 1)[0] for file in file_names]

# Import signals
sig_id_ord = [re.split('_|\.', os.path.split(sig_dir)[-1], 1)[0] for sig_dir in sig_dirs]
sigs_import = [np.load(os.path.join(sig_dir, 'extractedsignals.npy')) for sig_dir in sig_dirs]
if 'Water' in task:
    sig_frame_num = 1500
    sig_frame_max = sig_start + sig_frame_num
else:
    sig_frame_max = np.min([x.shape[1] for x in sigs_import])
    sig_frame_num = sig_frame_max - sig_start


#%% Format data

behav_frame_num = np.min([etho[0].shape[0] for etho in etho_out])
data = [interp_missing(etho[0][:behav_frame_num, :].T).T for etho in deepcopy(etho_out)]  # deepcopy(etho_out) makes another copy of etho_out
labels = [label.translate(None, '()<>/-').replace(' ', '_') for label in etho_out[0][1]]  # Assumes all labels are the same
time_ix = labels.index(time_label)
behav_fps = int(1 / np.diff(np.squeeze(data[0][:, time_ix])).mean())

# Correct time in EPM data
# Experiments had a 10-s delay before recording and calcium imaging
if task == 'EPM':
    data_frame_num = [[]] * file_num
    for subj in np.arange(file_num):
        data[subj][:, time_ix] -= 10
        pre_rec_ix = np.where(data[subj][:, time_ix] < 0)[0]  # Index of time points prior to recording
        data[subj] = np.delete(data[subj], pre_rec_ix, axis=0)
        data_frame_num[subj] = behav_frame_num - len(pre_rec_ix)
else:
    data_frame_num = [behav_frame_num for _ in np.arange(file_num)]

# Normalize behavioral data
# Set range of values from -1 to 1. Exclude timestamps (first 2 columns).
data_norm = list(data)
for subj in np.arange(file_num):
    var_maxes = np.max(np.absolute(data_norm[subj][:, 2:]), axis=0, keepdims=True)
    zero_max_mask = var_maxes == 0      # Don't want to divide by zero\
    var_maxes[zero_max_mask] = 1        # Set max as 1 instead
    data_norm[subj][:, 2:] = data_norm[subj][:, 2:] / \
                             np.repeat(var_maxes, data_frame_num[subj], axis=0)

# Truncate signal and events so all have the same number of frames
sigs_list = [sig[:, sig_start:sig_frame_max] for sig in sigs_import]

# Number of neurons per subject
num_cells_per_subj = [x.shape[0] for x in sigs_list]
cell_subj_id = [np.array(num_subj_cells * [sig_id]) for num_subj_cells, sig_id in zip(num_cells_per_subj, sig_id_ord)]
num_cells = sum(num_cells_per_subj)

# Reshape array of signal into one matrix
sigs = np.concatenate(tuple(sigs_list), axis=0)
sigs_id = np.concatenate(tuple(cell_subj_id), axis=0)
sig_ts = np.arange(sig_frame_num, dtype=float) / sig_fps

# Normalize signal
if event_data:
    cell_std = np.std(sigs, axis=1, keepdims=True)
    sigs_z = sigs / cell_std.repeat(sig_frame_num,axis=1)
else:
    sigs_z = map(stats.zscore, sigs)
    sigs_z = np.column_stack(sigs_z).T  # WATCH OUT FOR THE TRANSPOSE!!!


#%% Designate excited-by- & inhibited-by-TMT cells

if task in ['TMT', 'PNT']:
    tmt_exc_cells_import = np.loadtxt(tmt_exc_file, dtype=np.int32)
    tmt_inh_cells_import = np.loadtxt(tmt_inh_file, dtype=np.int32)
    
    #subj_cell_num_import = np.loadtxt(subj_cell_file, dtype='str')
    #first_cell_ix = np.cumsum(subj_cell_num_import[:, 1].astype(int))
    
    # Same order for TMT actually...
    tmt_exc_cells = tmt_exc_cells_import
    tmt_inh_cells = tmt_inh_cells_import


#%% Create new behavioral data

data_norm2 = list(data_norm)  # copy list

if task == 'EPM':
    for d in np.arange(len(data_norm2)):
        
        # Make it work for list of data..
        open_col = labels.index('In_zoneOpen_arms__centerpoint')
        closed_col = labels.index('In_zoneClosed_arms__centerpoint')
        center_d_col = labels.index('Distance_to_zone')
        
        ##  Separate distance from center by arm ##
        center_d_closed = data_norm2[d][:, center_d_col] * data_norm2[d][:, closed_col]
        center_d_open = data_norm2[d][:, center_d_col] * data_norm2[d][:, open_col]
        
        ## Create arm transitions ##
        # Identify state transitions
        # Index corresponds to the first behavioral frame of transition.
        open_diff = np.diff(np.squeeze(data_norm2[d][:, open_col]))
        close_diff = np.diff(np.squeeze(data_norm2[d][:, closed_col]))
        
        # Behavioral frame of change
        close_enter_mask = np.append(False, close_diff == 1)
        close_exit_mask = np.append(False, close_diff == -1)
        open_enter_mask = np.append(False, open_diff == 1)
        open_exit_mask = np.append(False, open_diff == -1)
        close_enter_frames = np.where(close_enter_mask)[0] + 1
        close_exit_frames = np.where(close_exit_mask)[0] + 1
        open_enter_frames = np.where(open_enter_mask)[0] + 1
        open_exit_frames = np.where(open_exit_mask)[0] + 1
        
        close_exit_num = len(close_exit_frames)
        open_enter_num = len(open_enter_frames)

        # Find closed-to-open transitions
        close2open = np.zeros(data_frame_num[d])
        for frame in close_exit_frames:
        
            # Frame of closed-arm entry
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
                next_close_enter_frame = data_frame_num[d]
            
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
                next_open_exit_frame = data_frame_num[d]
        
            # Must meet certain criteria to be classified as transition:
            # 1. Have been in closed arm for certain period of time (arm0_t_min)
            # 2. Must traverse center zone quickly (center_t_limit)
            # 3. Be in open arm for certain period of tie (arm1_t_min)
            if next_open_enter_frame < next_close_enter_frame and\
               frame - close_enter_frame > arm0_t_min * behav_fps and\
               next_open_enter_frame - frame < center_t_limit * behav_fps and\
               next_open_exit_frame - next_open_enter_frame > arm1_t_min * behav_fps:
                frame_start = frame - arm0_t_min * behav_fps
                frame_end = next_open_enter_frame + arm1_t_min * behav_fps
                close2open[frame_start:frame_end] = 1

        # Find open-to-closed transitions
        open2close = np.zeros(data_frame_num[d])
        for frame in open_exit_frames:
        
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
                next_open_enter_frame = data_frame_num[d]
            
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
                next_close_exit_frame = data_frame_num[d]
        
            # Must meet certain criteria to be classified as transition:
            # 1. Have been in closed arm for certain period of time (arm0_t_min)
            # 2. Must traverse center zone quickly (center_t_limit)
            # 3. Be in open arm for certain period of tie (arm1_t_min)
            if next_close_enter_frame < next_open_enter_frame and\
               frame - open_enter_frame > arm0_t_min * behav_fps and\
               next_close_enter_frame - frame < center_t_limit * behav_fps and\
               next_close_exit_frame - next_close_enter_frame > arm1_t_min * behav_fps:
                frame_start = frame - arm0_t_min * behav_fps
                frame_end = next_close_enter_frame + arm1_t_min * behav_fps
                open2close[frame_start:frame_end] = 1
        
        # Add new data
        data_norm2[d] = np.column_stack((data_norm2[d], center_d_closed,
                                                        center_d_open,
                                                        close_enter_mask,
                                                        close_exit_mask,
                                                        open_enter_mask,
                                                        open_exit_mask,
                                                        close2open,
                                                        open2close))
    
    # Add labels of new variables
    labels_final = labels + ['center_distance_in_closed_arm',
                             'center_distance_in_open_arm',
                             'closed_arm_entrance',
                             'closed_arm_exit',
                             'open_arm_entrance',
                             'open_arm_exit',
                             'closed_to_open_transition',
                             'open_to_closed_transition']
else:
    labels_final = labels


#%% Downsample to match calcium imaging time
# NOTE: data_ds is var x time (row-major order)

data_ds = []

for d in np.arange(file_num):
   # Find matching "bin" in calcium imaging time for each time point of behavioral data
   bin_ix = np.digitize(data_norm2[d][:, time_ix], sig_ts)
   
   data_ds.append(np.nan * np.zeros((len(labels_final), sig_frame_num), dtype=float))
   # Downsample
   for dbin in np.arange(sig_frame_num):
       bin_pts = np.where(bin_ix == dbin+1)[0]
       if bin_pts.size:
           data_ds[d][:, dbin] = np.mean(data_norm2[d][bin_pts, :], axis=0)
    

#%% GLM - Clean and select data

# Choose variables
if task == 'EPM':
    model_vars = [['Velocity',
                   'In_zoneOpen_arms__centerpoint',
                   'In_zoneClosed_arms__centerpoint',
                   'closed_to_open_transition',
                   'open_to_closed_transition',
                   'center_distance_in_closed_arm',
                   'center_distance_in_open_arm'],      # Model 1
                  
                  ['Velocity',
                   'In_zoneOpen_arms__centerpoint',
                   'In_zoneClosed_arms__centerpoint',
                   'closed_to_open_transition',
                   'open_to_closed_transition'],        # Model 2
                   
                   ['Velocity',
                   'In_zoneOpen_arms__centerpoint',
                   'In_zoneClosed_arms__centerpoint',
                   'center_distance_in_closed_arm',
                   'center_distance_in_open_arm'],      # Model 3
                   
                   ['Velocity',
                   'closed_to_open_transition',
                   'open_to_closed_transition',
                   'center_distance_in_closed_arm',
                   'center_distance_in_open_arm']]      # Model 4
else:
    model_vars = [['Velocity',
                   'Mobility_stateHighly_mobile',
                   'Mobility_stateImmobile',
                   'Distance_to_point']]

# Create formula
if event_data:
    formula_base = 'signal ~ '
else:
    formula_base = 'signal ~ signal_tminus1 + signal_tminus2 + '
formulas = [formula_base  + ' + '.join(var_set) for var_set in model_vars]
num_vars = [len(formula.split(' + ')) + 1 for formula in formulas] # add constant term
num_models = len(formulas)

# Find time points with bad data
#valid_frames = [[]] * file_num
#
#for d in np.arange(len(data_ds)):
#    # Remove time points with missing data
#    valid_frames[d] = np.where(np.isfinite(data_ds[d][time_ix, :]))[0]
    
    # Select behavioral variables
#    valid_behav_ix = np.ones(len(selected_behavs.shape, dtype=bool))
#    print "Behaviral variables chosen are:"
#    for n, behav in enumerate(selected_behavs):
#        if np.any(data_ds[d][valid_frames[d], labels_final.index(behav)]):
#            print " + " + behav
#        else:
#            print " - " + behav + ": chosen but does not vary, thus omitted"
#            valid_behav_ix[n] = False


#%% Logistic regression

Y, X = dmatrices(formulas[0], df, return_type="dataframe")

model = linear_model.LogisticRegression()
model = model.fit(X, y.astype(bool))


#%% GLM - Model

create_plots = True

if create_plots:
    plt.ioff()
    plot_dir = os.path.join(save_dir, 'plots')
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

p_vals = [np.zeros((num_cells, ncol)) for ncol in num_vars]
coeffs = [np.zeros((num_cells, ncol)) for ncol in num_vars]
r_sq = [np.zeros((num_cells, 2)) for _ in np.arange(num_models)]
aic = [np.zeros(num_cells) for _ in np.arange(num_models)]

for cell in np.arange(num_cells):
    sig_subj_id = sigs_id[cell]
    subj_ix = behav_id_ord.index(sig_subj_id)
    
    # Behavioral data
    df = pd.DataFrame(data=data_ds[subj_ix].T, columns=labels_final)
#    df = sm.add_constant(df)

    # Setup variables    
    y = sigs_z[cell, :]
    df['signal'] = y
    if not event_data:
        tminus1 = df['signal'][:-1]
        tminus2 = df['signal'][:-2]
        tminus1_norm = tminus1 / np.max(np.absolute(tminus1))
        tminus2_norm = tminus2 / np.max(np.absolute(tminus2))
        df['signal_tminus1'] = np.concatenate((np.nan * np.ones(1), tminus1_norm))
        df['signal_tminus2'] = np.concatenate((np.nan * np.ones(2), tminus2_norm))
    
    # Run OLS
    for m, formula in enumerate(formulas):
        model = smf.ols(data=df, formula=formula).fit()  # first two frames has nan for signal_tminus2
        p_vals[m][cell, :] = model.pvalues
        coeffs[m][cell, :] = model.params
        r_sq[m][cell, :] = [model.rsquared, model.rsquared_adj]
        aic[m][cell] = model.aic
        
        # Plot
        if create_plots:
            prstd, iv_l, iv_u = wls_prediction_std(model)
        
            fig, ax  = plt.subplots(2, 1)
            fig.suptitle("Model {}: cell {}".format(m+1, cell))
            
            ax[0].scatter(df.index, df['signal'], label="data")
            ax[0].plot(model.fittedvalues, 'r-')
            ax[0].plot(iv_l, 'r--')
            ax[0].plot(iv_u, 'r--')
            ax[0].set_ylabel("Model (& observed) fluorescence")
            ax[0].xaxis.set_visible(False)
        
            ax[1].plot(df['signal'] - model.fittedvalues, 'g-')
            ax[1].set_ylabel("Fluoresence residual")
            ax[1].set_xlabel("Time")
            ax[0].set_xlim(ax[1].get_xlim())
        
            plt.savefig(os.path.join(plot_dir, "model{}_cell{}.png".format(m+1, cell)),
                        dpi=200, bbox_inches='tight', transparent=True)
            plt.close()

# Create array for single AIC for each model
aic_all = np.zeros(num_models)

for m in np.arange(num_models):
    
    # Calculate AIC of model for ALL neurons
    aic_all[m] = np.sum(aic[m]) - 2*num_vars[m]*(num_cells-1)

    # Save data
    np.savetxt(os.path.join(save_dir, 'aic{}.txt'.format(m+1)),
               aic[m], delimiter='\t')
    np.savetxt(os.path.join(save_dir, 'r_sq{}.txt'.format(m+1)),
               r_sq[m], delimiter='\t')
    np.savetxt(os.path.join(save_dir, 'p_values{}.txt'.format(m+1)),
               p_vals[m], delimiter='\t')
    np.savetxt(os.path.join(save_dir, 'coeffs{}.txt'.format(m+1)),
               coeffs[m], delimiter='\t')
    formula_file = open(os.path.join(save_dir, 'formula{}.txt'.format(m+1)), 'w')
    formula_file.write(formulas[m])
    formula_file.close()

np.savetxt(os.path.join(save_dir, 'aic_whole_model.txt'), aic_all)


#%% GLM - Identify neurons significantly affected by each variable

# Get flattened index of signficant variables for each neuron
sig_ix = [bh_correction(model_ps.flatten()) for model_ps in p_vals]

# Get i, j coordinates
# ie, get neuron and variable that is significant
sig_ij = [np.unravel_index(model_sig_ix, model_ps.shape)\
          for model_sig_ix, model_ps in zip(sig_ix, p_vals)]

# Mask of significant variables for each neuron
sig_mask = [np.zeros(model_ps.shape, dtype=bool) for model_ps in p_vals]
for m, ij in enumerate(sig_ij):
    sig_mask[m][ij] = True
#    sig_mask[sig_i, sig_j] = True

# Mask of positively and negatively significant variables
pos_sig_mask = [np.logical_and(mask, model_coeffs > 0) for mask, model_coeffs in zip(sig_mask, coeffs)]
neg_sig_mask = [np.logical_and(mask, model_coeffs < 0) for mask, model_coeffs in zip(sig_mask, coeffs)]

# Mask of significant varialbes with +1 and -1 for positve and negative
# relationship
dir_mask = [np.zeros(model_coeffs.shape, dtype=int) for model_coeffs in coeffs]
for m, (pos, neg) in enumerate(zip(pos_sig_mask, neg_sig_mask)):
    dir_mask[m][pos] = 1
    dir_mask[m][neg] = -1

pos_ct = [model_mask.sum(axis=0) for model_mask in pos_sig_mask]
neg_ct = [model_mask.sum(axis=0) for model_mask in neg_sig_mask]

exc_pos_ct = [model_mask[tmt_exc_cells].sum(axis=0) for model_mask in pos_sig_mask]
exc_neg_ct = [model_mask[tmt_exc_cells].sum(axis=0) for model_mask in neg_sig_mask]
inh_pos_ct = [model_mask[tmt_inh_cells].sum(axis=0) for model_mask in pos_sig_mask]
inh_neg_ct = [model_mask[tmt_inh_cells].sum(axis=0) for model_mask in neg_sig_mask]
 
for m in np.arange(num_models):
    variables = ['intercept'] + re.split(' ~ | \+ ', formulas[m])[1:]
    
    print "Model {}:".format(m+1)
    print "All cells ({})".format(num_cells)
    for v, var in enumerate(variables):
        print "\t{} cells are signficantly affected by {} in the positive direction".format(pos_ct[m][v], var)
        print "\t{} cells are signficantly affected by {} in the negative direction".format(neg_ct[m][v], var)
    print "TMT-excited cells ({})".format(len(tmt_exc_cells))
    for v, var in enumerate(variables):
        print "\t{} cells are signficantly affected by {} in the positive direction".format(exc_pos_ct[m][v], var)
        print "\t{} cells are signficantly affected by {} in the negative direction".format(exc_neg_ct[m][v], var)
    print "TMT-inhibited cells ({})".format(len(tmt_inh_cells))
    for v, var in enumerate(variables):
        print "\t{} cells are signficantly affected by {} in the positive direction".format(inh_pos_ct[m][v], var)
        print "\t{} cells are signficantly affected by {} in the negative direction".format(inh_neg_ct[m][v], var)

    # Save data
    np.savetxt(os.path.join(save_dir, 'summary_glm_all_cells_model{}.txt'.format(m+1)),
               np.row_stack((pos_ct, neg_ct)).T, delimiter='\t')
    np.savetxt(os.path.join(save_dir, 'summary_glm__tmt_exc_cells_model{}.txt'.format(m+1)),
               np.row_stack((exc_pos_ct, exc_neg_ct)).T, delimiter='\t')
    np.savetxt(os.path.join(save_dir, 'summary_glm__tmt_inh_cells_model{}.txt'.format(m+1)),
               np.row_stack((inh_pos_ct, inh_neg_ct)).T, delimiter='\t')


#%% GLM - Venn diagram
model = 1
var1 = 0
var2 = 3
var3 = 4
variables = re.split(' ~ | \+ ', formulas[model])[1:]

set1 = set(np.where(pos_sig_mask[model][:, var1])[0].astype(str))
set2 = set(np.where(pos_sig_mask[model][:, var2])[0].astype(str))
set3 = set(np.where(pos_sig_mask[model][:, var3])[0].astype(str))

venn3([set1, set2, set3], (variables[var1], variables[var2], variables[var3]))
plt.show()

#%% GLM - Plot coefficients
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['blue', 'gray', 'red']
ref = 11 # color coded by closed to open

for cell_response in [-1, 0, 1]:
    mask = dir_mask[:, ref] == cell_response
    x0 = coeffs[mask, 11] # closed to open
    x1 = coeffs[mask, 12] # open to closed
    x2 = coeffs[mask, 13] # distance into open arm
    
    ax.scatter(x0, x1, x2, c=colors[cell_response+1], s=50)

ax.set_xlabel("Closed to open")
ax.set_ylabel("Open to closed")
ax.set_zlabel("Distance into open arm")
plt.show()


#%% MORE EPM ANALYSIS BELOW
#%% Re-organize

pre_frame_num = 25
post_frame_num = 25
frame_dur = 0.2  # Time between frame starts

# Index variables
close2open_ix = labels_final.index('closed_to_open_transition')
open2close_ix = labels_final.index('open_to_closed_transition')
from_close_ix = labels_final.index('closed_arm_exit')
from_open_ix = labels_final.index('open_arm_exit')
to_close_ix = labels_final.index('closed_arm_entrance')
to_open_ix = labels_final.index('open_arm_entrance')


#%% Event focused - Create windows for arm transitions for one animal (for EPM)

# Array of transition data from all animals
# (transition type) x (cells) x (time in transition frame)
pre_sig_trial_avg  = np.nan * np.ones((2, num_cells, pre_frame_num))
post_sig_trial_avg = np.nan * np.ones((2, num_cells, pre_frame_num))
cell_bookmark = 0  # Indicates next cell to fill in above arrays

for subj in np.arange(len(sig_id_ord)):
    id = sig_id_ord[subj]
    subj_sig_ix = sigs_id == id
    sigs_subj = sigs_z[subj_sig_ix, :]
    num_subj_cells, num_frames = sigs_subj.shape
    
    subj_data_ix = behav_id_ord.index(id)
    data_subj = data_ds[subj]
    
    # Index frames
    close2open_frames      = data_subj[close2open_ix, :] > 0
    open2close_frames      = data_subj[open2close_ix, :] > 0
    close_exit_frames_all  = data_subj[from_close_ix, :] > 0
    open_exit_frames_all   = data_subj[from_open_ix, :] > 0
    close_enter_frames_all = data_subj[to_close_ix, :] > 0
    open_enter_frames_all  = data_subj[to_open_ix, :] > 0
    
    # Events during transition
    # eg, close_exit_frames are frames where subject exits closed arm during 
    # closed-to-open arm transitions
    close_exit_frames  = np.logical_and(close_exit_frames_all,  close2open_frames)
    open_exit_frames   = np.logical_and(open_exit_frames_all,   open2close_frames)
    close_enter_frames = np.logical_and(close_enter_frames_all, open2close_frames)
    open_enter_frames  = np.logical_and(open_enter_frames_all,  close2open_frames)
    
    # Gather data
    exit_frames  = [close_exit_frames, open_exit_frames]
    enter_frames = [open_enter_frames, close_enter_frames]

    # Go through each transition
    for tt in np.arange(2):
        # Determine if animal made any transitions
        if np.any(exit_frames[tt]):
            # Create windows around arm transitions
            pre_sig = np.stack([sigs_subj[:, ix-pre_frame_num:ix] for ix in np.where(exit_frames[tt])[0]], axis=-1)
            post_sig = np.stack([sigs_subj[:, ix:ix+post_frame_num] for ix in np.where(enter_frames[tt])[0]], axis=-1)
            
            # Average time series response
            pre_sig_trial_avg[tt, cell_bookmark:cell_bookmark+num_subj_cells, :] = \
                pre_sig.mean(axis=2)
            post_sig_trial_avg[tt, cell_bookmark:cell_bookmark+num_subj_cells, :] = \
                post_sig.mean(axis=2)
                
    cell_bookmark += num_subj_cells


#%%
transition   = ["Closed-to-open", "Open-to-closed"]

for tt in np.arange(2):
    # Average epoch response per cell
    pre_sig_epoch_avg = np.nanmean(pre_sig_trial_avg[tt], axis=-1)
    post_sig_epoch_avg = np.nanmean(post_sig_trial_avg[tt], axis=-1)
    
    # Plot
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 3)
    cell = 0   # example cell to plot
    color_pre = 'lightsteelblue'
    color_post = 'steelblue'
    c_palette = sns.color_palette([color_pre, color_post])
    
    # Plot data from all cells (averaged over trials)
    x0, y0 = np.meshgrid(np.arange(0, -(pre_frame_num+1), -1) * frame_dur,
                         np.arange(num_cells+1))
    x1, y1 = np.meshgrid(np.arange(post_frame_num+1) * frame_dur,
                         np.arange(num_cells+1))
    max_z = np.nanmax(np.concatenate((pre_sig_trial_avg[tt].flatten(),
                                      post_sig_trial_avg[tt].flatten())))
    min_z = np.nanmin(np.concatenate((pre_sig_trial_avg[tt].flatten(),
                                      post_sig_trial_avg[tt].flatten())))
    
    ax_pre_cmap = plt.subplot(gs[0, 0])
    ax_pre_cmap.set_title("{} transitions".format(transition[tt]))
    im_pre = ax_pre_cmap.pcolormesh(x0, y0, pre_sig_trial_avg[tt])
    im_pre.set_clim([min_z, max_z])
    ax_pre_cmap.set_ylabel("Cells")
    ax_pre_cmap.set_xlabel('Pre')
    
    ax_post_cmap = plt.subplot(gs[0, 1])
    im_post = ax_post_cmap.pcolormesh(x1, y1, post_sig_trial_avg[tt])
    im_post.set_clim([min_z, max_z])
    ax_post_cmap.yaxis.set_visible(False)
    ax_post_cmap.set_xlabel("Post")
    
    # Plot average
    x0 = np.arange(0, -pre_frame_num, -1) * frame_dur
    x1 = np.arange(post_frame_num) * frame_dur
    y0 = np.nanmean(pre_sig_trial_avg[tt], axis=0)
    y1 = np.nanmean(post_sig_trial_avg[tt], axis=0)
    e0 = stats.sem(pre_sig_trial_avg[tt], axis=0, nan_policy='omit')
    e1 = stats.sem(post_sig_trial_avg[tt], axis=0, nan_policy='omit')
    ymax = np.amax(np.concatenate((y0, y1))) + np.amax(np.concatenate((e0, e1)))
    ymin = np.amin(np.concatenate((y0, y1))) - np.amin(np.concatenate((e0, e1)))
    
    ax_pre_sig = plt.subplot(gs[1, 0])
    ax_pre_sig.plot(x0, y0, color_pre)
    ax_pre_sig.fill_between(x0, y0-e0, y0+e0,
                            facecolor=color_pre, alpha=0.3)
    ax_pre_sig.set_ylabel('Fluorescence value')
    ax_pre_sig.set_ylim(ymin, ymax)    
    ax_pre_sig.set_xlabel('Pre')
    
    ax_post_sig = plt.subplot(gs[1, 1])
    ax_post_sig.plot(x1, y1, color_post)
    ax_post_sig.fill_between(x1, y1-e1, y1+e1,
                            facecolor=color_post, alpha=0.3)
    ax_post_sig.set_ylim(ymin, ymax)
    ax_post_sig.yaxis.set_visible(False)
    ax_post_sig.set_xlabel('Post')
    
    # Plot average pre/post
    w = 0.8
    x = np.arange(2)
    y = [np.nanmean(pre_sig_epoch_avg),
         np.nanmean(post_sig_epoch_avg)]
    e = [stats.sem(pre_sig_epoch_avg, nan_policy='omit'),
         stats.sem(pre_sig_epoch_avg, nan_policy='omit')]
    
    ax_avg_sig = plt.subplot(gs[1, 2])
    sns.swarmplot(np.repeat(x, num_cells),
                  np.concatenate([pre_sig_epoch_avg, post_sig_epoch_avg]),
                  ax=ax_avg_sig,
                  palette=c_palette)
    error_config = {'ecolor': '0.3'}
    ax_avg_sig.bar(x-w/2, y, yerr=e, error_kw=error_config, fill=False)
    ax_avg_sig.set_title("Average response")
    ax_avg_sig.set_ylim(ymin, ymax)
    ax_avg_sig.yaxis.set_visible(False)
    ax_avg_sig.set_xticklabels(('Pre', 'Post'))

    gs.tight_layout(fig)

    # Save data
    if event_data:
        data_type = "events"
    else:
        data_type = "traces"
    np.savetxt(os.path.join(save_dir, '{}_pre_transitions_{}.txt'.format(transition[tt], data_type)),
               pre_sig_trial_avg[tt], delimiter = '\t')        
    np.savetxt(os.path.join(save_dir, '{}_post_transitions_{}.txt'.format(transition[tt], data_type)),
               post_sig_trial_avg[tt], delimiter = '\t')
    plt.savefig(os.path.join(save_dir, '{}_{}.svg'.format(transition[tt], data_type)),
                bbox_inches='tight', transparent=True)


#%% Calculate preference index
# Based on events

open_col = labels.index('In_zoneOpen_arms__centerpoint')
closed_col = labels.index('In_zoneClosed_arms__centerpoint')

pref_index_avg = [[]] * file_num
pref_index_avg_p = [[]] * file_num
for f in np.arange(file_num):
#    cell_ix = np.where(sigs_id == sig_id_ord[f])[0]
    closed_frames = data_ds[f][closed_col, :] > 0
    open_frames = data_ds[f][open_col, :] > 0
    
#    sig_temp = sigs_animal[cell_ix, :]
    
    # Signal values during open (or closed) arm only
    sig_closed = sigs_animal[:, closed_frames]
    sig_open = sigs_animal[:, open_frames]
    
    # Preference index
    # FIND NEW FORMULA!!!!!!!!!!
    pref_index_avg[f] = np.log(sig_open.mean(axis=1)/sig_closed.mean(axis=1))
    pref_index_avg_p[f] = np.array([stats.mannwhitneyu(x, y)[1] for x, y in zip(sig_closed, sig_open)])

    np.savetxt(os.path.join(save_dir, 'preference_indices_by_avg_{}.txt'.format(sig_id_ord[animal])),
               np.concatenate(pref_index_avg), delimiter='\t')
    np.savetxt(os.path.join(save_dir, 'preference_indices_by_avg_p_{}.txt'.format(sig_id_ord[animal])),
               np.concatenate(pref_index_avg_p), delimiter = '\t')


#%% Create activity map

x = data_ds[0][2, :]
y = data_ds[0][3, :]
c, _, [fig, ax, im] = activity_map(x, y, sigs_animal[3, :], binsize=0.025, plot=True, sigma=2, cmap='jet')

im.set_clim([0, 1])
fig.canvas.draw()