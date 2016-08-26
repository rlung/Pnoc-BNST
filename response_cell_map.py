import numpy as np
import scipy.ndimage as img
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import os

#%% Import data

# Directory with files
os.chdir('/data/Dropbox (Stuber Lab)/Python/Jose/Odor_TMT/new stuff/Activity Map')

# Get response data
with open('ave_zscores_tmt.txt') as intxt:
    signal = [[float(numStr) for numStr in line.strip().split(',')] for line in intxt]
signal = np.array(signal)

response = signal
cellNum = len(response)

# Get TIFF files in cell image directory
imgDir = 'footprints/*.tif'
imgFiles = glob.glob(imgDir)
imgNum = len(imgFiles)

# Convert files to numpy array
cellMaps = np.array([img.imread(imgFile) for imgFile in imgFiles])

# Checkpoint
if imgNum != cellNum:
    print 'FAILED CHECKPOINT: Number of cells in signal data does not match number of image files'


#%% Create map

# Create composite map
cell_maps_resp = np.nan * np.zeros(cellMaps.shape)
cellMapAll = np.nan * np.zeros_like(cellMaps[0,:,:],dtype=np.float64)

for m in np.arange(cellNum):
    cellMap = cellMaps[m, ...].astype(np.float64)
    cellMap = img.filters.median_filter(cellMap, size=(3,3))  # smooth image
    cellIX = np.nonzero(cellMap)
    cellMax = cellMap[cellIX].max()
    
    cell_map_resp = cellMap/cellMax * response[m]
    thresh = 0.3 * response[m]
    cell_map_resp_thresh = np.where(np.absolute(cell_map_resp) > np.absolute(thresh), cell_map_resp, 0)
    
    cell_maps_resp[m, ...] = cell_map_resp_thresh
    cellMapAll = np.where(cell_map_resp_thresh != 0, cell_map_resp_thresh, cellMapAll)  # linear scale
    cellMapAll = np.where(cell_map_resp_thresh != 0, np.log(cell_map_resp_thresh), cellMapAll)  # log scale


#%% Plot

c = np.ma.array(cellMapAll, mask=np.isnan(cellMapAll))

cmap = cm.seismic
cmap.set_bad('k', 0)
im = plt.pcolormesh(c, vmin=-5, vmax=5, cmap=cmap)
plt.colorbar()