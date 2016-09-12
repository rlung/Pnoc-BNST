import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from bokeh.plotting import figure, output_notebook, show
import os
import re
import glob

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

# Working directory
os.chdir(os.path.join(base_dir, 'PNOC'))

# Recording parameters
frame_max = 4496                            # Number of frames to truncate data at
fps = 5                                     # Number of frames in a second
min_frame = 60*fps                          # Number of frames in a minute **NOT user-defined**
base_frame = min_frame * 1                  # Number of frames during baseline period
bin_frame = 1                               # Number of frames per bin
min_bin = min_frame / bin_frame             # Number of bins per minute **NOT user-difined**
response_bin = min_bin * 1                  # Number of bins during response period
response_frames = response_bin * bin_frame  # Number of frames during response period **NOT user-defined**

# Directories
sig_dirs = glob.glob('PNOC Traces_Odor/PNOC*')
ev_dirs = glob.glob('PNOC Events_Odor/PNOC*')

# Plotting parameters
sns.set_context('talk')
color_ctrl = 'gray'
color_tmt = 'darkorange'
color_exc = 'red'
color_inh = 'blue'


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
#ev_epochs = np.stack(())
num_cells = sig_epochs.shape[ax_cell]
num_frames = sig_epochs.shape[ax_time]

# Data before and after water and TMT stimuli (stimulus x neuron x time)
sig_base = sig_epochs[:2, :, -base_frame:]
sig_post = sig_epochs[1:, :, :]
sig_data = np.concatenate((sig_base, sig_post), axis=ax_time)

#ev_base = ev_epochs[:2, :, -base_frame:]
#ev_post = ev_epochs[1:, :, :]
#ev_data = np.concatenate((ev_base, ev_post), axis=ax_time)

# Bin data
# Number of frames in 'base' and 'post' MUST be multiple of 'bin_frames'
data_binned_raw = sig_data.reshape((2, num_cells, -1, bin_frame)).mean(axis=3)
num_bins = data_binned_raw.shape[ax_time]

# Calculate z scores
base_bin = base_frame / bin_frame
data_binned_z = zscore_base(data_binned_raw, base_bin, axis=2)

# Count event rate (per min)
#events_rates = np.concatenate(ev_data[:, :, :base_bin].mean(axis=ax_time) * min_frame,
#                              ev_data[:, :, base_bin:].mean(axis=ax_time) * min_frame,
#                              axis=ax_time)                                             # **CHECK THIS**

#%% Plot data

x, y = np.meshgrid(np.arange(num_bins+1)-base_bin, np.arange(num_cells+1))
dy, dx = x.shape
max_z = np.max(data_binned_z.flatten())

# As colorplot
# **First dimension (x) is plotted along y-axis with pcolor**
fig, ax  = plt.subplots(2, 1)

titles= ["Water response", "TMT response"]

for i in range(2):
    ax[i].set_title(titles[i], fontsize=20)
    ax[i].set_ylabel("Neurons")
    
    im = ax[i].pcolormesh(x, y, data_binned_z[i, :, :], cmap='jet')
    ax[i].set_xlim(np.array([0, dx-1]) - base_bin)
    ax[i].set_ylim([0, num_cells])
    im.set_clim([0, max_z])
    
    ax[i].axvline(x=0, linestyle='--', color='w')

cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Calcium z-score")


#%% Average data

# Average response trace (stimulus x bins)
data_trace_avg_z = data_binned_z.mean(axis=ax_cell)
data_trace_sem_z = stats.sem(data_binned_z, axis=ax_cell)

# Average response during response windows pre and post stimulus (for both 
# stimuli) for each neuron (stimulus x neuron x pre/post)
data_response_avg_z = np.stack((data_binned_z[:, :, base_bin-response_bin:base_bin].mean(axis=ax_time), 
                                data_binned_z[:, :, base_bin:base_bin+response_bin].mean(axis=ax_time)),
                               axis=ax_time)


#%% Plot averages

# Average traces
ax_trace = plt.subplot2grid((1, 4), (0, 0), colspan=3)

x = np.arange(num_bins) - base_bin + 1
y = data_trace_avg_z
e = data_trace_sem_z

plt.plot(x, y[0, :], 'gray', x, y[1, :], 'seagreen')
plt.fill_between(x, y[0, :]-e[0, :], y[0, :]+e[0, :], facecolor='gray', alpha=0.3)
plt.fill_between(x, y[1, :]-e[1, :], y[1, :]+e[1, :], facecolor='seagreen', alpha=0.3)
ax_trace.set_title("Average trace response to stimulus", fontsize=20)
ax_trace.set_ylabel("Calcium z score", fontsize=20)
ax_trace.set_xlabel("Time from stimulus", fontsize=20)
plt.xlim([x[0], x[-1]])

# Average response
ax_bar = plt.subplot2grid((1, 4), (0, 3))
w = 0.3
left = 0.2
x = np.arange(2) + left

y = data_response_avg_z.mean(axis=ax_cell)
e = stats.sem(data_response_avg_z, axis=ax_cell)

error_config = {'ecolor': '0.3'}

ax_bar.bar(x, y[0, :], w, color='gray', yerr=e[0, :], error_kw=error_config)
ax_bar.bar(x+w, y[1, :], w, color='seagreen', yerr=e[1, :], error_kw=error_config)

ax_bar.set_title("Average response\nduring response windows", fontsize=20)
ax_bar.set_xlim([0, 2])
ax_bar.set_xticks(np.arange(2) + 0.5)
ax_bar.set_xticklabels(('Pre', 'Post'), fontsize=20)

# Average events


#%% Distribution of average responses

h2o_response_distro = np.histogram(data_response_avg_z[0, :, 1], bins=50, range=(-4, 10))
tmt_response_distro = np.histogram(data_response_avg_z[1, :, 1], bins=50, range=(-4, 10))

# As cumulative distribution
plt.figure()
y0 = np.cumsum(h2o_response_distro[0], dtype='float64')/num_cells
y1 = np.cumsum(tmt_response_distro[0], dtype='float64')/num_cells
x0 = h2o_response_distro[1][:-1]
x1 = tmt_response_distro[1][:-1]
plt.plot(x0, y0, 'gray', x1, y1, 'seagreen')
plt.title("Distribution of responses to stimuli (Water vs TMT)", fontsize=30)
plt.ylabel("Cumulative proportion", fontsize=20)
plt.xlabel("Response (z score)", fontsize=20)

# As histogram
plt.figure()
y0 = h2o_response_distro[0]
y1 = tmt_response_distro[0]
x0 = h2o_response_distro[1][:-1]
x1 = tmt_response_distro[1][:-1]
plt.fill_between(x0, np.zeros(y0.shape), y0, facecolor='gray', alpha=0.3)
plt.fill_between(x1, np.zeros(y1.shape), y1, facecolor='seagreen', alpha=0.3)
plt.title("Distribution of responses to stimuli (Water vs TMT)", fontsize=30)
plt.ylabel("Cumulative proportion", fontsize=20)
plt.xlabel("Response (z score)", fontsize=20)


#%% Classify neurons

# Determine statistically signficant differences from stimulus
u_vals = np.zeros((2, num_cells))
p_vals = np.zeros((2, num_cells))
for stimulus in range(2):
    for neuron in range(num_cells):
        u_vals[stimulus, neuron], p_vals[stimulus, neuron] = stats.mannwhitneyu(data_binned_z[stimulus, neuron, base_bin-response_bin:base_bin],
                                                                                data_binned_z[stimulus, neuron, base_bin:base_bin+response_bin])
p_05_ij = np.where(p_vals < 0.05)

alpha = 0.05
sig_diff_ix = bh_correction(p_vals.flatten(), alpha=alpha)  # **flatten() goes down last axis first (opposite of Matlab)

# Classify as excitatory or inhibitory
# Excitatory neurons are those with an increase in response from pre- and post-stimulus
# AND a significant difference between calcium response pre- and post-stimulus. Inhibitory
# are similarly defined.
exc_ij = np.where(data_response_avg_z[:, :, 1] > data_response_avg_z[:, :, 0])
inh_ij = np.where(data_response_avg_z[:, :, 1] < data_response_avg_z[:, :, 0])
exc_ix = np.ravel_multi_index(exc_ij, (2, num_cells))
inh_ix = np.ravel_multi_index(inh_ij, (2, num_cells))
sig_exc_ix = sig_diff_ix[np.in1d(sig_diff_ix, exc_ix)]
sig_inh_ix = sig_diff_ix[np.in1d(sig_diff_ix, inh_ix)]
sig_exc_ij = np.unravel_index(sig_exc_ix, (2, num_cells))
sig_inh_ij = np.unravel_index(sig_inh_ix, (2, num_cells))

h2o_exc_cells = sig_exc_ij[1][np.where(sig_exc_ij[0] == 0)[0]]
h2o_inh_cells = sig_inh_ij[1][np.where(sig_inh_ij[0] == 0)[0]]
tmt_exc_cells = sig_exc_ij[1][np.where(sig_exc_ij[0] == 1)[0]]
tmt_inh_cells = sig_inh_ij[1][np.where(sig_inh_ij[0] == 1)[0]]

num_h2o_exc_cells = len(np.where(sig_exc_ij[0] == 0)[0])
num_h2o_inh_cells = len(np.where(sig_inh_ij[0] == 0)[0])
num_tmt_exc_cells = len(np.where(sig_exc_ij[0] == 1)[0])
num_tmt_inh_cells = len(np.where(sig_inh_ij[0] == 1)[0])

# Print output
print "Found {0} neurons with p value less than {1} to water stimulus.".format(len(np.where(p_05_ij[0] == 0)[0]), alpha)
print "{0} neurons passed Benjamini-Hochberg correction.".format(len(np.where(sig_diff_ix < num_cells)[0]))
print "{0}/{1} excitatory and {2}/{3} inhibitory neurons classified.".format(num_h2o_exc_cells, len(np.where(exc_ij[0] == 0)[0]),
                                                                             num_h2o_inh_cells, len(np.where(inh_ij[0] == 0)[0]))
print ""
print "Found {0} neurons with p value less than {1} to TMT stimulus.".format(len(np.where(p_05_ij[0] == 1)[0]), alpha)
print "{0} neurons passed Benjamini-Hochberg correction.".format(len(np.where(sig_diff_ix < num_cells)[0]))
print "{0}/{1} excitatory and {2}/{3} inhibitory neurons classified.".format(num_tmt_exc_cells, len(np.where(exc_ij[0] == 1)[0]),
                                                                             num_tmt_inh_cells, len(np.where(inh_ij[0] == 1)[0]))


#%% Average by classification

# Average response trace split into groups: excitatory and inhibitory to each stimuli
h2o_exc_trace_avg_z = data_binned_z[:, h2o_exc_cells, :].mean(axis=ax_cell)
h2o_inh_trace_avg_z = data_binned_z[:, h2o_inh_cells, :].mean(axis=ax_cell)
tmt_exc_trace_avg_z = data_binned_z[:, tmt_exc_cells, :].mean(axis=ax_cell)
tmt_inh_trace_avg_z = data_binned_z[:, tmt_inh_cells, :].mean(axis=ax_cell)

# SEM
h2o_exc_trace_sem_z = stats.sem(data_binned_z[:, h2o_exc_cells, :], axis=ax_cell)
h2o_inh_trace_sem_z = stats.sem(data_binned_z[:, h2o_inh_cells, :], axis=ax_cell)
tmt_exc_trace_sem_z = stats.sem(data_binned_z[:, tmt_exc_cells, :], axis=ax_cell)
tmt_inh_trace_sem_z = stats.sem(data_binned_z[:, tmt_inh_cells, :], axis=ax_cell)

# Average response during response windows pre and post stimulus (for both 
# stimuli) for each neuron (stimulus x neuron x pre/post)
h2o_exc_response_avg_z = data_response_avg_z[:, h2o_exc_cells, :]
h2o_inh_response_avg_z = data_response_avg_z[:, h2o_inh_cells, :]
tmt_exc_response_avg_z = data_response_avg_z[:, tmt_exc_cells, :]
tmt_inh_response_avg_z = data_response_avg_z[:, tmt_inh_cells, :]


#%% Plot averages by classification

# Average traces - TMT exc during both stimuli
plt.figure()
ax_trace = plt.subplot2grid((1, 4), (0, 0), colspan=3)

x = np.arange(num_bins) - base_bin
y1 = tmt_exc_trace_avg_z[h2o, :]
y2 = tmt_exc_trace_avg_z[tmt, :]
e1 = tmt_exc_trace_sem_z[h2o, :]
e2 = tmt_exc_trace_sem_z[tmt, :]

plt.plot(x, y1, color_ctrl, x, y2, color_exc)
plt.xlim([x[0], x[-1]])

plt.fill_between(x, y1-e1, y1+e1, facecolor=color_ctrl, alpha=0.3)
plt.fill_between(x, y2-e2, y2+e2, facecolor=color_exc, alpha=0.3)

# Average response
ax_bar = plt.subplot2grid((1, 4), (0, 3))
w = 0.3
left = 0.2
x = np.arange(2) + left

y = tmt_exc_response_avg_z.mean(axis=ax_cell)
y0 = y[0, :]
y1 = y[1, :]
e = stats.sem(tmt_inh_response_avg_z, axis=ax_cell)
e0 = e[0, :]
e1 = e[1, :]

error_config = {'ecolor': '0.3'}

plt.bar(x, y0, w, color=color_ctrl, yerr=e0, error_kw=error_config)
plt.bar(x+w, y1, w, color=color_exc, yerr=e1, error_kw=error_config)
ax_bar.set_xlim([0, 2])
ax_bar.set_xticks(np.arange(2) + 0.5)
ax_bar.set_xticklabels(('Baseline', 'Stimulus'))

# Calcium events
plt.figure()

all_cells = np.arange(num_cells)
tmt_ne_cells = all_cells[~np.in1d(all_cells, sig_diff_ix - num_cells)]  # subtract by num_cells since tmt index adds num_cells when flattened

w = 0.2
x0 = np.arange(2)
x1 = np.arange(2) + w
x2 = np.arange(2) + 2*w
y0 = event_rates[1, tmt_ne_cells, :]
y1 = event_rates[1, tmt_exc_cells, :]
y2 = event_rates[1, tmt_inh_cells, :]
e0 = stats.sem(event_rates[1, tmt_ne_cells, :])
e1 = stats.sem(event_rates[1, tmt_exc_cells, :]
e2 = stats.sem(event_rates[1, tmt_inh_cells, :]
error_config = {'ecolor': '0.3'}

plt.figure()
plt.bar(x0, y0, w, color=color_ctrl, yerr=e0, error_kw=error_config)
plt.bar(x1, y1, w, color=color_exc, yerr=e1, error_kw=error_config)
plt.bar(x2, y2, w, color=color_inh, yerr=2, error_kw=error_config)
ax_bar.set_xlim([0, 2])
ax_bar.set_xticks(np.arange(2) + 0.5)
ax_bar.set_xticklabels(('Baseline', 'Stimulus'))

# Distribution
# Distribution (cumulative histogram) of average responses - TMT excited
h2o_response_distro = np.histogram(tmt_inh_response_avg_z[0, :, 1], bins=50, range=(-4, 10))
tmt_response_distro = np.histogram(tmt_inh_response_avg_z[1, :, 1], bins=50, range=(-4, 10))

y0 = np.cumsum(h2o_response_distro[0], dtype='float64')/tmt_inh_response_avg_z.shape[1]
y1 = np.cumsum(tmt_response_distro[0], dtype='float64')/tmt_inh_response_avg_z.shape[1]
x0 = h2o_response_distro[1][:-1]
x1 = tmt_response_distro[1][:-1]

plt.figure()
plt.plot(x0, y0, color_ctrl, x1, y1, color_exc)
plt.title("Distribution of response to stimuli (Water vs TMT) in TMT-excited cells", fontsize=30)
plt.ylabel("Cumulative proportion", fontsize=20)
plt.xlabel("Repsponse (z score)", fontsize=20)