import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from bokeh.plotting import figure, output_notebook, show
import os

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

# Recording parameters
fps = 5
min_frame = 60*fps
epoch_frame = min_frame*fps
bin_frame = 1
min_bin = min_frame / bin_frame

# Plotting parameters
pylab.rcParams['figure.figsize'] = 16, 9
sns.set_style('dark')
color_ctrl = 'gray'
color_tmt = 'seagreen'
color_exc = 'cadetblue'


#%% Import data

all = np.array([])
count = 0
for path, subdirs, files in os.walk('PNOC Traces_Odor'):
    for name in files:
        data = np.load(os.path.join(path, name))
        data = data[:, :4496]
        if all.any(): all = np.append(all, data, axis=0)
        else: all = data

num_cells, num_frames = all.shape

# Print output
print "{0} cells found with {1} frames.".format(num_cells, num_frames)


#%% Organize data

base_frame = min_frame * 1

ax_stim = 0
ax_cell = 1
ax_time = 2

h2o = 0
tmt = 1

epochs = np.stack((all[:, :1450], all[:, 1499:2949], all[:, 2998:4448]), axis=ax_stim)
num_cells = epochs.shape[ax_cell]
num_frames = epochs.shape[ax_time]

# Data before and after water and TMT stimuli (stimulus x neuron x time)
base = epochs[:2, :, -base_frame:]
post = epochs[1:, :, :]
data = np.concatenate((base, post), axis=ax_time)

# Bin data
# Number of frames in 'base' and 'post' MUST be multiple of 'bin_frames'
data_binned_raw = data.reshape((2, num_cells, -1, bin_frame))
# if not np.array_equal(data_binned_raw[1, 4, :, :].flatten(), data[1, 4, :]): print "Binning error"
data_binned_raw = data_binned_raw.mean(axis=3)
num_bins = data_binned_raw.shape[ax_time]

# Calculate z scores
base_bin = base_frame / bin_frame
base_binned_avg = data_binned_raw[:, :, :base_bin].mean(axis=ax_time, keepdims=True)
base_binned_std = data_binned_raw[:, :, :base_bin].std(axis=ax_time, keepdims=True)
data_binned_z = (data_binned_raw - base_binned_avg.repeat(num_bins, axis=ax_time)) / base_binned_std.repeat(num_bins, axis=ax_time)


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

# As line plots
x = np.arange(num_bins) - base_bin + 1
y = data_binned_z
ybnd = [np.floor(y.min()), np.ceil(y.max())]

fig, ax  = plt.subplots(2, 1)
titles = ["Water response", "TMT response"]
colors = ['gray', 'seagreen']

for i in range(2):
    for c in range(num_cells):
        ax[i].plot(x, y[i, c, :], colors[i], alpha=0.3)
        ax[i].axvline(x=0, linestyle='--', color='w')
        
        ax[i].set_title(titles[i], fontsize=20)
        ax[i].set_ylabel("Calcium z score")
        ax[i].set_ylim(ybnd)
        ax[i].set_xlim([x[0], x[-1]])


#%% Average data

response_bin = min_bin * 1

# Average response trace (stimulus x bins)
data_trace_avg_z = data_binned_z.mean(axis=ax_cell)
data_trace_sem_z = stats.sem(data_binned_z, axis=ax_cell)

# Average response during response windows pre and post stimulus (for both 
# stimuli) for each neuron (stimulus x neuron x pre/post)
data_response_avg_z = np.stack((data_binned_z[:, :, base_bin-response_bin:base_bin].mean(axis=ax_time), 
                                data_binned_z[:, :, base_bin:base_bin+response_bin].mean(axis=ax_time)),
                               axis=ax_time)
# data_response_sem_z = np.stack((stats.sem(data_binned_z[:, :, base_bin-response_bin:base_bin], axis=ax_time), 
#                                 stats.sem(data_binned_z[:, :, base_bin:response_bin], axis=ax_time)),
#                                axis=ax_time)


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


#%% Distribution of average responses

h2o_response_distro = np.histogram(data_response_avg_z[0, :, 1], bins=50, range=(-4, 10))
tmt_response_distro = np.histogram(data_response_avg_z[1, :, 1], bins=50, range=(-4, 10))

# As cumulative distribution
y0 = np.cumsum(h2o_response_distro[0], dtype='float64')/num_cells
y1 = np.cumsum(tmt_response_distro[0], dtype='float64')/num_cells
x0 = h2o_response_distro[1][:-1]
x1 = tmt_response_distro[1][:-1]
plt.plot(x0, y0, 'gray', x1, y1, 'seagreen')
plt.title("Distribution of responses to stimuli (Water vs TMT)", fontsize=30)
plt.ylabel("Cumulative proportion", fontsize=20)
plt.xlabel("Response (z score)", fontsize=20)

# As histogram
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
sig_diff_ix = bh_correction(p_vals.flatten(), alpha=alpha)

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
ax_trace = plt.subplot2grid((1, 4), (0, 0), colspan=3)

x = np.arange(num_bins) - base_bin
y1 = tmt_inh_trace_avg_z[h2o, :]
y2 = tmt_inh_trace_avg_z[tmt, :]
e1 = tmt_inh_trace_sem_z[h2o, :]
e2 = tmt_inh_trace_sem_z[tmt, :]

plt.plot(x, y1, color_ctrl, x, y2, color_exc)
plt.xlim([x[0], x[-1]])

plt.fill_between(x, y1-e1, y1+e1, facecolor=color_ctrl, alpha=0.3)
plt.fill_between(x, y2-e2, y2+e2, facecolor=color_exc, alpha=0.3)

# Average response
ax_bar = plt.subplot2grid((1, 4), (0, 3))
w = 0.3
left = 0.2
x = np.arange(2) + left

y = tmt_inh_response_avg_z.mean(axis=ax_cell)
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

# Distribution
# Distribution (cumulative histogram) of average responses - TMT excited
h2o_response_distro = np.histogram(tmt_inh_response_avg_z[0, :, 1], bins=50, range=(-4, 10))
tmt_response_distro = np.histogram(tmt_inh_response_avg_z[1, :, 1], bins=50, range=(-4, 10))

y0 = np.cumsum(h2o_response_distro[0], dtype='float64')/tmt_inh_response_avg_z.shape[1]
y1 = np.cumsum(tmt_response_distro[0], dtype='float64')/tmt_inh_response_avg_z.shape[1]
x0 = h2o_response_distro[1][:-1]
x1 = tmt_response_distro[1][:-1]
plt.plot(x0, y0, color_ctrl, x1, y1, color_exc)
plt.title("Distribution of response to stimuli (Water vs TMT) in TMT-excited cells", fontsize=30)
plt.ylabel("Cumulative proportion", fontsize=20)
plt.xlabel("Repsponse (z score)", fontsize=20)