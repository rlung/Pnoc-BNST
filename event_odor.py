import numpy as np
from scipy import stats
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from bokeh.plotting import figure, output_notebook, show
import os
import re
import glob
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
    from my_functions import zscore_base, bh_correction
else:
    print "my_function not imported. Directory not identified."


#%% Parameters

# Define if events or fluorescence data
event_data = False
odor = 'TMT'

# User-defined behavior parameters
frame_max = 4496                            # Number of frames to truncate data at
fps = 5                                     # Number of frames in a second
base_time = 60                              # Length of bins (s)
bin_frame = fps * 1                         # Number of frames per bin

# Recording parameters
min_frame = fps * 60                        # Number of frames in a minute **NOT user-defined**
min_bin = min_frame / bin_frame             # Number of bins per minute **NOT user-difined**
base_frame = fps * base_time                # Number of frames during baseline period
base_bin = base_frame / bin_frame           # Number of bins during baseline
response_bin = min_bin * 1                  # Number of bins during response period
response_frames = response_bin * bin_frame  # Number of frames during response period **NOT user-defined**

# Directories
os.chdir(os.path.join(base_dir, 'PNOC'))
sig_dirs = glob.glob('PNOC_{0}/PNOC_{0}_Traces/PNOC*'.format(odor))

# Directory to save data
now = datetime.now()
save_dir = 'data/{}_odor-'.format(odor) + now.strftime('%y%m%d-%H%M%S')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Plotting parameters
sns.set_context('talk')
color_ctrl = 'gray'
if odor == 'TMT':
    color_stim = 'darkorange'
else:
    color_stim = 'green'
color_exc = 'red'
color_inh = 'blue'
cmap = 'coolwarm'
fs_ax = 10
fs_fig = 15
gauss_win = signal.gaussian(15, 2.5)
window = gauss_win / np.sum(gauss_win)


#%% Import data

# Import signal
sig_id_ord = [re.split('_|\.', os.path.split(sig_dir)[-1], 1)[0] for sig_dir in sig_dirs]
sigs_import = [np.load(os.path.join(sig_dir, 'extractedsignals.npy')) for sig_dir in sig_dirs]
sig_frame_num = np.min([x.shape[1] for x in sigs_import])
sig_file_num = len(sigs_import)

# Import events
#ev_id_ord = [re.split('_|\.', os.path.split(ev_dir)[-1], 1)[0] for ev_dir in ev_dirs]
#evs_import = [np.load(os.path.join(ev_dir, 'extractedsignals.npy')) for ev_dir in ev_dirs]      # *** Make sure directory is correct **
#ev_frame_num = np.min([x.shape[1] for x in sigs_import])
#ev_file_num = len(evs_import)

# Check
#if sig_frame_num != ev_frame_num:
#    raise UserWarning("Number of frames between signal and event files do not match.")
#if sig_file_num != ev_file_num:
#    raise UserWarning("Number of files between signal and event files do not match.")

# Truncate signal and events so all have the same number of frames
sigs_list = [sig[:, :frame_max] for sig in sigs_import]
#evs_list = [ev[:, :frame_max] for ev in evs_import]

# Number of neurons per subject
num_cells_per_subj = [x.shape[0] for x in sigs_list]
#if [x.shape[0] for x in evs_list] != num_cells_per_subj:
#    raise UserWarning("Number of cells in each file is not consistent between signal and events.")
cell_subj_id = [np.array(num_cells * [sig_id]) for num_cells, sig_id in zip(num_cells_per_subj, sig_id_ord)]

# Reshape arrays into one matrix
sigs = np.concatenate(sigs_list, axis=0)
sigs_id = np.concatenate(cell_subj_id, axis=0)
sig_ts = np.arange(sig_frame_num, dtype=float) / fps

#events = np.concatenate(evs_list, axis=0)

# Print output
print "{} cells found from {} subjects.".format(sum(num_cells_per_subj), sig_file_num)


#%% Organize data

ax_stim = 0
ax_cell = 1
ax_time = 2

h2o = 0
tmt = 1

sig_epochs = np.stack((sigs[:, :1450], sigs[:, 1499:2949], sigs[:, 2998:4448]), axis=ax_stim)
num_cells = sig_epochs.shape[ax_cell]
num_frames = sig_epochs.shape[ax_time]

# Data before and after water and TMT stimuli (stimulus x neuron x time)
sig_base = sig_epochs[:2, :, -base_frame:]
sig_post = sig_epochs[1:, :, :]
sig_data = np.concatenate((sig_base, sig_post), axis=ax_time)

# Bin data
# Number of frames in 'base' and 'post' MUST be multiple of 'bin_frames'
sig_binned_raw = sig_data.reshape((2, num_cells, -1, bin_frame)).mean(axis=3)
num_bins = sig_binned_raw.shape[ax_time]

# Normalize without mean 0
#if event_data:
# Keep "0 at 0": all values remain positive if event data
cell_std = np.std(np.concatenate((sig_binned_raw[0, ...],
                                  sig_binned_raw[1, ...]), axis=1),
                  axis=1, keepdims=True)
sig_binned_z = sig_binned_raw / np.tile(cell_std[np.newaxis, :], (2, 1, num_bins))
print "Normalized keeping min 0"
#else:
#    base_bin = base_frame / bin_frame
#    sig_binned_z = zscore_base(sig_binned_raw, base_bin, axis=2)
#    print "Normalized by z score to baseline"

# Shift data so baseline is 0
# Not really necessary if already doing zscore
base_avg = sig_binned_z[:, :, :base_bin].mean(axis=ax_time, keepdims=True)
sig_base0 = sig_binned_z - np.repeat(base_avg, num_bins, axis=ax_time)

# Average time seires trace (stimulus x bins)
sig_time_series_avg = sig_base0.mean(axis=ax_cell)
sig_time_series_sem = stats.sem(sig_base0, axis=ax_cell)

# Average response during response windows pre and post stimulus (for both 
# stimuli) for each neuron (stimulus x neuron x pre/post)
sig_response_avg = sig_base0[:, :, base_bin:base_bin+response_bin].mean(axis=ax_time)

# Save data
np.savetxt(os.path.join(save_dir, 'data_w_base_0_h2o.txt'),
           sig_base0[0, ...], delimiter='\t')
np.savetxt(os.path.join(save_dir, 'data_w_base_0_{}.txt'.format(odor)),
           sig_base0[1, ...], delimiter='\t')
np.savetxt(os.path.join(save_dir, 'data_response_avg_h2o.txt'),
           sig_response_avg[0, ...], delimiter='\t')
np.savetxt(os.path.join(save_dir, 'data_response_avg_{}.txt'.format(odor)),
           sig_response_avg[1, ...], delimiter='\t')


#%% Plot: all data

x, y = np.meshgrid(np.arange(num_bins+1)-base_bin, np.arange(num_cells+1))
dy, dx = x.shape
max_z = 5
#max_z = np.max(sig_binned_z)

# As colorplot
# **First dimension (x) is plotted along y-axis with pcolor**
fig, ax  = plt.subplots(1, 2)

titles= ["Water response", "{} response".format(odor)]

order = np.argsort(sig_response_avg[1, :])

for i in range(2):
    ax[i].set_title(titles[i], fontsize=fs_fig)
    if i == 0: ax[i].set_ylabel("Neurons", fontsize=fs_ax)
    
    im = ax[i].pcolormesh(x, y, sig_base0[i, order, :], cmap=cmap)
    ax[i].set_xlim(np.array([0, dx-1]) - base_bin)
    ax[i].set_ylim([0, num_cells])
    im.set_clim([-max_z, max_z])
    
    ax[i].axvline(x=0, linestyle='--', linewidth=0.5, color='k')
    
    # Save raw data
    np.savetxt(os.path.join(save_dir, '{}_raw_response_stim{}.txt'.format(odor, i)),
               sig_base0[i, order, :], delimiter='\t')

cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Calcium signal")

plt.savefig(os.path.join(save_dir, '{}_raw_response.svg'.format(odor)),
            bbox_inches='tight', transparent=True)


#%% Plot: averages

# Average traces
plt.figure()
ax_trace = plt.subplot2grid((1, 5), (0, 0), colspan=3)

x = np.arange(num_bins) - base_bin + 1
y0 = np.concatenate((np.convolve(sig_time_series_avg[0, :base_bin], window, mode='same'),
                     np.convolve(sig_time_series_avg[0, base_bin:], window, mode='same')), axis=0)
y1 = np.concatenate((np.convolve(sig_time_series_avg[1, :base_bin], window, mode='same'),
                     np.convolve(sig_time_series_avg[1, base_bin:], window, mode='same')), axis=0)
e0 = np.concatenate((np.convolve(sig_time_series_sem[0, :base_bin], window, mode='same'),
                     np.convolve(sig_time_series_sem[0, base_bin:], window, mode='same')), axis=0)
e1 = np.concatenate((np.convolve(sig_time_series_sem[1, :base_bin:], window, mode='same'),
                     np.convolve(sig_time_series_sem[1, base_bin:], window, mode='same')), axis=0)

ax_trace.plot(x, y0, color_ctrl, label="Water")
ax_trace.plot(x, y1, color_stim, label=odor)
ax_trace.fill_between(x, y0-e0, y0+e0, facecolor=color_ctrl, alpha=0.3)
ax_trace.fill_between(x, y1-e1, y1+e1, facecolor=color_stim, alpha=0.3)
ax_trace.set_title("Average time series to stimulus", fontsize=fs_fig)
ax_trace.set_ylabel("Calcium signal", fontsize=fs_ax)
ax_trace.set_xlabel("Time from stimulus", fontsize=fs_ax)
ax_trace.set_xlim([x[0], x[-1]])

# Average response
ax_bar = plt.subplot2grid((1, 5), (0, 3))
w = 0.8

x = np.arange(2) - w/2
x0 = x[0]
x1 = x[1]
y = sig_response_avg[:, :].mean(axis=ax_cell)
y0 = y[0]
y1 = y[1]
e = stats.sem(sig_response_avg[:, :], axis=ax_cell)
e0 = e[0]
e1 = e[1]

#sns.swarmplot(np.repeat(x, num_cells),
#              sig_response_avg.flatten(),
#              ax=ax_bar,
#              palette=sns.color_palette([color_ctrl, color_stim]))
error_config = {'ecolor': '0.3'}
ax_bar.bar(x0, y0, w, color=color_ctrl, yerr=e0, error_kw=error_config)
ax_bar.bar(x1, y1, w, color=color_stim, yerr=e1, error_kw=error_config)

ax_bar.set_title("Average response\nduring response windows", fontsize=fs_fig)
ax_bar.yaxis.set_visible(False)
ax_bar.set_xticks(x + w/2)
ax_bar.set_xticklabels(('Water', odor), fontsize=fs_ax)

# Set axis limits equal
lim_trace = ax_trace.get_ylim()
lim_bar = ax_bar.get_ylim()
lim_new = [np.min((lim_trace[0], lim_bar[0])), np.max((lim_trace[1], lim_bar[1]))]
ax_trace.set_ylim(lim_new)
ax_bar.set_ylim(lim_new)

# Average response - percent change from baseline
#ax_bar = plt.subplot2grid((1, 5), (0, 4))
#w = 0.6
#x = np.arange(2) - w/2
#
#percent_change = sig_response_avg[:, :, 1] /\
#                 sig_response_avg[:, :, 0]
#y = percent_change.mean(axis=ax_cell)
#e = stats.sem(percent_change, axis=ax_cell)
#
#error_config = {'ecolor': '0.3'}
#
#sns.swarmplot(np.repeat(x, num_cells),
#              percent_change.flatten(),
#              ax=ax_bar,
#              palette=sns.color_palette([color_ctrl, color_tmt]))
#bars = ax_bar.bar(x, y, w, yerr=e, error_kw=error_config, fill=None)
#bars[0].set_edgecolor(color_ctrl)
#bars[1].set_edgecolor(color_tmt)
#
#ax_bar.set_title("Average response\nduring response windows", fontsize=fs_fig)
#ax_bar.set_ylabel("Change from baseline (proportion)", fontsize=fs_ax)
#ax_bar.yaxis.tick_right()
#ax_bar.yaxis.set_label_position("right")
##ax_bar.set_xlim([0, 2])
##ax_bar.set_xticks(np.arange(2) + 0.5)
#ax_bar.set_xticklabels(('Water', 'TMT'), fontsize=fs_ax)

# Save data
plt.savefig(os.path.join(save_dir, '{}_avg_response.svg'.format(odor)),
            bbox_inches='tight', transparent=True)


#%% Plot: distribution of average responses

# By percent change (use for events normalized with "0 as 0")
#h2o_response_distro = np.histogram(percent_change[0, :], bins=50, range=(-4, 10))
#tmt_response_distro = np.histogram(percent_change[1, :], bins=50, range=(-4, 10))

# By average response value
h2o_response_distro = np.histogram(sig_response_avg[0, :], bins=50, range=(-4, 10))
tmt_response_distro = np.histogram(sig_response_avg[1, :], bins=50, range=(-4, 10))

# Plot
fig, ax = plt.subplots(2)
x0 = h2o_response_distro[1][:-1]
x1 = tmt_response_distro[1][:-1]

# As cumulative distribution
y0 = np.cumsum(h2o_response_distro[0], dtype='float64')/num_cells
y1 = np.cumsum(tmt_response_distro[0], dtype='float64')/num_cells
ax[0].plot(x0, y0, color_ctrl)
ax[0].plot(x1, y1, color_stim)
ax[0].set_title("Distribution of responses to stimuli (Water vs TMT)", fontsize=fs_fig)
ax[0].set_ylabel("Cumulative proportion", fontsize=fs_ax)
ax[0].xaxis.set_visible(False)

# As histogram
y0 = h2o_response_distro[0]
y1 = tmt_response_distro[0]
ax[1].fill_between(x0, np.zeros(y0.shape), y0, facecolor=color_ctrl, alpha=0.3)
ax[1].fill_between(x1, np.zeros(y1.shape), y1, facecolor=color_stim, alpha=0.3)
ax[1].set_ylabel("Cumulative proportion", fontsize=fs_ax)
ax[1].set_xlabel("Response score", fontsize=fs_ax)

plt.savefig(os.path.join(save_dir, 'odor_response_distro_all.svg'),
            bbox_inches='tight', transparent=True)


#%% Classify neurons

# Determine statistically signficant differences from stimulus
u_vals = np.zeros((2, num_cells))
p_vals = np.zeros((2, num_cells))
for stimulus in range(2):
    for neuron in range(num_cells):
#        u_vals[stimulus, neuron], p_vals[stimulus, neuron] = stats.mannwhitneyu(sig_binned_z[stimulus, neuron, base_bin-response_bin:base_bin],
#                                                                                sig_binned_z[stimulus, neuron, base_bin:base_bin+response_bin])
        u_vals[stimulus, neuron], p_vals[stimulus, neuron] = stats.wilcoxon(sig_base0[stimulus, neuron, base_bin:base_bin+response_bin])
p_05_ij = np.where(p_vals < 0.05)

alpha = 0.05
sig_diff_ix = bh_correction(p_vals.flatten(), alpha=alpha)  # **flatten() goes down last axis first (opposite of Matlab)

# Classify as excitatory or inhibitory
# Excitatory neurons are those with an increase in response from pre- and post-stimulus
# AND a significant difference between calcium response pre- and post-stimulus. Inhibitory
# are similarly defined.
exc_ij = np.where(sig_response_avg[:, :] > 0)
inh_ij = np.where(sig_response_avg[:, :] < 0)
exc_ix = np.ravel_multi_index(exc_ij, (2, num_cells))
inh_ix = np.ravel_multi_index(inh_ij, (2, num_cells))
sig_exc_ix = sig_diff_ix[np.in1d(sig_diff_ix, exc_ix)]
sig_inh_ix = sig_diff_ix[np.in1d(sig_diff_ix, inh_ix)]
sig_exc_ij = np.unravel_index(sig_exc_ix, (2, num_cells))
sig_inh_ij = np.unravel_index(sig_inh_ix, (2, num_cells))

#h2o_exc_cells = sig_exc_ij[1][np.where(sig_exc_ij[0] == 0)[0]]
#h2o_inh_cells = sig_inh_ij[1][np.where(sig_inh_ij[0] == 0)[0]]
tmt_exc_cells = sig_exc_ij[1][np.where(sig_exc_ij[0] == 1)[0]]
tmt_inh_cells = sig_inh_ij[1][np.where(sig_inh_ij[0] == 1)[0]]

#num_h2o_exc_cells = len(np.where(sig_exc_ij[0] == 0)[0])
#num_h2o_inh_cells = len(np.where(sig_inh_ij[0] == 0)[0])
num_tmt_exc_cells = len(np.where(sig_exc_ij[0] == 1)[0])
num_tmt_inh_cells = len(np.where(sig_inh_ij[0] == 1)[0])

# Print output
#print "Found {0} neurons with p value less than {1} to water stimulus.".format(len(np.where(p_05_ij[0] == 0)[0]), alpha)
#print "{0} neurons passed Benjamini-Hochberg correction.".format(len(np.where(sig_diff_ix < num_cells)[0]))
#print "{0}/{1} excitatory and {2}/{3} inhibitory neurons classified.".format(num_h2o_exc_cells, len(np.where(exc_ij[0] == 0)[0]),
#                                                                             num_h2o_inh_cells, len(np.where(inh_ij[0] == 0)[0]))
summary_str = "Found {0} neurons with p value less than {1} to TMT stimulus.\n".format(len(np.where(p_05_ij[0] == 1)[0]), alpha) +\
              "{0} neurons passed Benjamini-Hochberg correction.\n".format(len(np.where(sig_diff_ix < num_cells)[0])) +\
              "{0}/{1} excitatory and {2}/{3} inhibitory neurons classified.".format(num_tmt_exc_cells, len(np.where(exc_ij[0] == 1)[0]),
                                                                                     num_tmt_inh_cells, len(np.where(inh_ij[0] == 1)[0]))
print "\n" + summary_str
#%%
# Save data
summary_file = open(os.path.join(save_dir, 'summary.txt'), 'w')
summary_file.write(summary_str)
summary_file.close()

np.savetxt(os.path.join(save_dir, 'tmt_exc_cells.txt'),
           tmt_exc_cells, delimiter='\t')
np.savetxt(os.path.join(save_dir, 'tmt_inh_cells.txt'),
           tmt_inh_cells, delimiter='\t')
np.savetxt(os.path.join(save_dir, 'animal_cell_numbers.txt'),
           np.column_stack((sig_id_ord, num_cells_per_subj)), delimiter='\t', fmt='%s')


#%% Average by classification

# Average response trace split into groups: excitatory and inhibitory to each stimuli
#h2o_exc_trace_avg_z = sig_binned_z[:, h2o_exc_cells, :].mean(axis=ax_cell)
#h2o_inh_trace_avg_z = sig_binned_z[:, h2o_inh_cells, :].mean(axis=ax_cell)
tmt_exc_trace_avg_z = sig_base0[:, tmt_exc_cells, :].mean(axis=ax_cell)
tmt_inh_trace_avg_z = sig_base0[:, tmt_inh_cells, :].mean(axis=ax_cell)

#h2o_exc_trace_sem_z = stats.sem(sig_binned_z[:, h2o_exc_cells, :], axis=ax_cell)
#h2o_inh_trace_sem_z = stats.sem(sig_binned_z[:, h2o_inh_cells, :], axis=ax_cell)
tmt_exc_trace_sem_z = stats.sem(sig_base0[:, tmt_exc_cells, :], axis=ax_cell)
tmt_inh_trace_sem_z = stats.sem(sig_base0[:, tmt_inh_cells, :], axis=ax_cell)

# Average response during response windows pre and post stimulus (for both 
# stimuli) for each neuron (stimulus x neuron x pre/post)
#h2o_exc_response_avg_z = sig_response_avg[:, h2o_exc_cells, :]
#h2o_inh_response_avg_z = sig_response_avg[:, h2o_inh_cells, :]
tmt_exc_response_avg_z = sig_response_avg[:, tmt_exc_cells]
tmt_inh_response_avg_z = sig_response_avg[:, tmt_inh_cells]

## Average percent change
##h2o_exc_percent_change_avg = percent_change[:, h2o_exc_cells].mean(axis=ax_cell)
##h2o_inh_percent_change_avg = percent_change[:, h2o_inh_cells].mean(axis=ax_cell)
#tmt_exc_percent_change_avg = percent_change[:, tmt_exc_cells].mean(axis=ax_cell)
#tmt_inh_percent_change_avg = percent_change[:, tmt_inh_cells].mean(axis=ax_cell)
#
##h2o_exc_percent_change_sem = stats.sem(percent_change[:, h2o_exc_cells], axis=ax_cell)
##h2o_inh_percent_change_sem = stats.sem(percent_change[:, h2o_inh_cells], axis=ax_cell)
#tmt_exc_percent_change_sem = stats.sem(percent_change[:, tmt_exc_cells], axis=ax_cell)
#tmt_inh_percent_change_sem = stats.sem(percent_change[:, tmt_inh_cells], axis=ax_cell)

# Save data
np.savetxt(os.path.join(save_dir, '{}_exc_w_base_0_h2o.txt'.format(odor)),
           sig_base0[0, tmt_exc_cells, :], delimiter='\t')
np.savetxt(os.path.join(save_dir, '{}_exc_w_base_0_{}.txt'.format(odor, odor)),
           sig_base0[1, tmt_exc_cells, :], delimiter='\t')
np.savetxt(os.path.join(save_dir, '{}_exc_response_avg_h2o.txt'.format(odor)),
           tmt_exc_response_avg_z[0, ...], delimiter='\t')
np.savetxt(os.path.join(save_dir, '{}_exc_response_avg_{}.txt'.format(odor, odor)),
           tmt_exc_response_avg_z[1, ...], delimiter='\t')
np.savetxt(os.path.join(save_dir, '{}_inh_response_avg_h2o.txt'.format(odor)),
           tmt_inh_response_avg_z[0, ...], delimiter='\t')
np.savetxt(os.path.join(save_dir, '{}_inh_response_avg_{}.txt'.format(odor, odor)),
           tmt_inh_response_avg_z[1, ...], delimiter='\t')


#%% Plot averages by classification
# Lines with '#' at the end need to be modified to switch between excited and
# inhibited.

# Plotting parameters
color_resp = [color_exc, color_inh]
grp_cells = [tmt_exc_cells, tmt_inh_cells]
grp_trace_avg_z = [tmt_exc_trace_avg_z, tmt_inh_trace_avg_z]
grp_trace_sem_z = [tmt_exc_trace_sem_z, tmt_inh_trace_sem_z]
grp_response_avg_z = [tmt_exc_response_avg_z, tmt_inh_response_avg_z]

for gg in np.arange(len(grp_cells)):
    num_cells_in_class = len(grp_cells[gg]) #
    
    # Average traces - TMT exc during both stimuli
    plt.figure()
    ax_trace = plt.subplot2grid((1, 5), (0, 0), colspan=3)
    
    x = np.arange(num_bins) - base_bin
    y0 = np.concatenate((np.convolve(grp_trace_avg_z[gg][h2o, :base_bin], window, mode='same'),
                         np.convolve(grp_trace_avg_z[gg][h2o, base_bin:], window, mode='same')), axis=0)
    y1 = np.concatenate((np.convolve(grp_trace_avg_z[gg][tmt, :base_bin], window, mode='same'),
                         np.convolve(grp_trace_avg_z[gg][tmt, base_bin:], window, mode='same')), axis=0)
    e0 = np.concatenate((np.convolve(grp_trace_sem_z[gg][h2o, :base_bin], window, mode='same'),
                         np.convolve(grp_trace_sem_z[gg][h2o, base_bin:], window, mode='same')), axis=0)
    e1 = np.concatenate((np.convolve(grp_trace_sem_z[gg][tmt, :base_bin], window, mode='same'),
                         np.convolve(grp_trace_sem_z[gg][tmt, base_bin:], window, mode='same')), axis=0)

    ax_trace.plot(x, y0, color_ctrl, label="Water")
    ax_trace.plot(x, y1, color_resp[gg], label=odor)
    plt.fill_between(x, y0-e0, y0+e0, facecolor=color_ctrl, alpha=0.3)
    plt.fill_between(x, y1-e1, y1+e1, facecolor=color_resp[gg], alpha=0.3)
    ax_trace.set_ylabel("Normalized response")
    plt.xlim([x[0], x[-1]])

    # Average response
    ax_bar = plt.subplot2grid((1, 5), (0, 3))
    w = 0.8
    
    x = np.arange(2) - w/2
    x0 = x[0]
    x1 = x[1]
    y0 = grp_response_avg_z[gg].mean(axis=ax_cell)[h2o] #
    y1 = grp_response_avg_z[gg].mean(axis=ax_cell)[tmt] #
    e0 = stats.sem(grp_response_avg_z[gg], axis=ax_cell)[h2o] #
    e1 = stats.sem(grp_response_avg_z[gg], axis=ax_cell)[tmt] #
    
    #sns.swarmplot(np.repeat(x, num_cells_in_class),
    #              grp_response_avg_z[gg].flatten(), #
    #              ax=ax_bar,
    #              palette=sns.color_palette([color_ctrl, color_resp[gg]]))
    error_config = {'ecolor': '0.3'}
    plt.bar(x0, y0, w, color=color_ctrl, yerr=e0, error_kw=error_config)
    plt.bar(x1, y1, w, color=color_resp[gg], yerr=e1, error_kw=error_config)
    ax_bar.set_title("Average response\nduring response windows", fontsize=fs_fig)
    ax_bar.yaxis.set_visible(False)
    ax_bar.set_xticks(x + w/2)
    ax_bar.set_xticklabels(('Water', 'TMT'), fontsize=fs_ax)
    
    # Set axis limits equal
    lim_trace = ax_trace.get_ylim()
    lim_bar = ax_bar.get_ylim()
    lim_new = [np.min((lim_trace[0], lim_bar[0])), np.max((lim_trace[1], lim_bar[1]))]
    ax_trace.set_ylim(lim_new)
    ax_bar.set_ylim(lim_new)
    
    ## Average response - percent change from baseline
    #ax_bar_pc = plt.subplot2grid((1, 5), (0, 4))
    #w = 0.6
    #x = np.arange(2) - w/2
    #
    #percent_change = tmt_exc_response_avg_z[:, :, 1] /\
    #                 tmt_exc_response_avg_z[:, :, 0] #
    #y = percent_change.mean(axis=ax_cell)
    #e = stats.sem(percent_change, axis=ax_cell)
    #
    #sns.swarmplot(np.repeat(x, percent_change.shape[1]),
    #              percent_change.flatten(),
    #              ax=ax_bar_pc,
    #              palette=sns.color_palette([color_ctrl, color_resp[gg]]))
    #error_config = {'ecolor': '0.3'}
    #bars = ax_bar_pc.bar(x, y, w, yerr=e, error_kw=error_config, fill=None)
    #bars[0].set_edgecolor(color_ctrl)
    #bars[1].set_edgecolor(color_resp[gg])
    #
    #ax_bar_pc.set_title("Average response\nduring response windows", fontsize=fs_fig)
    #ax_bar_pc.set_ylabel("Percent change from baseline", fontsize=fs_ax)
    #ax_bar_pc.yaxis.tick_right()
    #ax_bar_pc.yaxis.set_label_position("right")
    #
    #ax_bar_pc.set_xticklabels(('Water', 'TMT'), fontsize=fs_ax)
    
    ## Calcium events
    #plt.figure()
    #
    #all_cells = np.arange(num_cells)
    #tmt_ne_cells = all_cells[~np.in1d(all_cells, sig_diff_ix - num_cells)]  # subtract by num_cells since tmt index adds num_cells when flattened
    #
    #w = 0.2
    #x0 = np.arange(2)
    #x1 = np.arange(2) + w
    #x2 = np.arange(2) + 2*w
    #y0 = event_rates[1, tmt_ne_cells, :]
    #y1 = event_rates[1, tmt_exc_cells, :]
    #y2 = event_rates[1, tmt_inh_cells, :]
    #e0 = stats.sem(event_rates[1, tmt_ne_cells, :])
    #e1 = stats.sem(event_rates[1, tmt_exc_cells, :])
    #e2 = stats.sem(event_rates[1, tmt_inh_cells, :])
    #error_config = {'ecolor': '0.3'}
    #
    #plt.figure()
    #plt.bar(x0, y0, w, color=color_ctrl, yerr=e0, error_kw=error_config)
    #plt.bar(x1, y1, w, color=color_exc, yerr=e1, error_kw=error_config)
    #plt.bar(x2, y2, w, color=color_inh, yerr=2, error_kw=error_config)
    #ax_bar.set_xlim([0, 2])
    #ax_bar.set_xticks(np.arange(2) + 0.5)
    #ax_bar.set_xticklabels(('Baseline', 'Stimulus'))
    
    # Distribution
    # Distribution (cumulative histogram) of average responses - TMT excited
    
    # By percent change (use for events normalized with "0 as 0")
    #h2o_response_distro = np.histogram(percent_change[0, :], bins=50, range=(-4, 10))
    #tmt_response_distro = np.histogram(percent_change[1, :], bins=50, range=(-4, 10))
    
    # By average response value
    h2o_response_distro = np.histogram(grp_response_avg_z[gg][0, :], bins=50, range=(-4, 10)) #
    tmt_response_distro = np.histogram(grp_response_avg_z[gg][1, :], bins=50, range=(-4, 10)) #
    
    y0 = np.cumsum(h2o_response_distro[0], dtype='float64')/h2o_response_distro[0].shape[0]
    y1 = np.cumsum(tmt_response_distro[0], dtype='float64')/tmt_response_distro[0].shape[0]
    x0 = h2o_response_distro[1][:-1]
    x1 = tmt_response_distro[1][:-1]
    
    fig, ax = plt.subplots()
    line_water, = ax.plot(x0, y0, color_ctrl, label='Water')
    line_tmt, = ax.plot(x1, y1, color_resp[gg], label='TMT')
    ax.set_title("Response distribution to stimuli (Water vs TMT)", fontsize=fs_fig)
    ax.set_ylabel("Cumulative proportion", fontsize=fs_ax)
    ax.set_xlabel("Response", fontsize=fs_ax)
    ax.legend(handles=[line_water, line_tmt], loc=4)
    
    plt.savefig(os.path.join(save_dir, 'odor_avg_response_tmt_exc.svg'),
                bbox_inches='tight', transparent=True)