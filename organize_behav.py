#!/usr/bin/env python

import numpy as np
import pandas as pd
import h5py as h5
import scipy.interpolate
import os
import glob
import multiprocessing as multi
from time import strftime
from datetime import datetime
from functools import partial
from argparse import ArgumentParser, RawTextHelpFormatter
import custom
import pupilize


def manage_data(data, n_cores=1, threshold=127, sample_period=200, key='cam/frames', time_limit=300000):
    '''Create pupil dataframe from file

    '''
    global df
    epoch_dict = {'1': 'base', '2': 'ctrl', '3': 'stim'}

    print("Analyzing {}...".format(os.path.basename(data)))

    _, animal_id, odor, order, plane, epoch = os.path.splitext(
        os.path.basename(data))[0].split('_')

    with h5.File(data, 'r') as hf:
        pupil_frames = hf[key]
        pupil_timestamps = hf[os.path.dirname(key) + '/timestamps']

        # Calculate pupil diameter
        pfunc = partial(pupilize.find_pupil, threshold=threshold, kernel=np.ones((7, 7)))
        if n_cores > 1:
            p = multi.Pool(processes=n_cores)
            boxes, _ = zip(*p.map(pfunc, pupil_frames))
        else:
            boxes, _ = zip(*map(pfunc, pupil_frames))
        pupil_diam = [w for _, _, w, _ in boxes]

        # Resample data
        # ts_max = min(time_limit, int(pupil_timestamps[-1]))
        # ts_new = np.arange(0, ts_max, sample_period)
        ts_new = np.array(df.index.levels[1])
        # pupil_resampled = custom.resample(pupil_diam, pupil_timestamps, ts_new, method=np.mean)
        pupil_resampled = scipy.interpolate.interp1d(pupil_timestamps, pupil_diam)(ts_new)

        behav = hf['behavior']
        trials = custom.resample(
            np.ones(behav['trials'].shape), behav['trials'], ts_new, method=np.any)
        rail_home = custom.resample(
            np.ones(behav['rail_home'].shape), behav['rail_home'], ts_new, method=np.any)
        rail_leave = custom.resample(
            np.ones(behav['rail_leave'].shape), behav['rail_leave'], ts_new, method=np.any)
        track = custom.resample(
            behav['track'][1], behav['track'][0], ts_new, method=np.sum)

    # Save data
    col_name = (animal_id, plane, order)
    # pdb.set_trace()
    df.set_value(epoch_dict[epoch], col_name + ('pupil', ), pupil_resampled)
    df.set_value(epoch_dict[epoch], col_name + ('trials', ), trials)
    df.set_value(epoch_dict[epoch], col_name + ('rail_home', ), rail_home)
    df.set_value(epoch_dict[epoch], col_name + ('rail_leave', ), rail_leave)
    df.set_value(epoch_dict[epoch], col_name + ('track', ), track)

    # df.set_value(ix0, col_name, pupil_resampled)
    # if df is None:
    #     series = {col_name: pupil_resampled}
    #     df = pd.DataFrame(
    #         series,
    #         index=pd.MultiIndex.from_tuples(
    #             zip([epoch] * len(ts_new), ts_new)
    #         )
    #     )
    # else:
    #     df[col_name, epoch] = pupil_resampled

    # save_file = '{}_{}_{}.txt'.format(animal_id, exp_day, epoch)
    # np.savetxt(save_file, pupil_diam)
    # print("Saved data to {}".format(save_file))

    print('Finished')


def main():
    parser = ArgumentParser(
        description="Calculate pupil size",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "data",
        help='HDF5 file (or directory containing files) with pupil data'
    )
    parser.add_argument(
        '-b', '--bin-size', default='200',
        help='Size of bins to resample pupil data'
    )
    parser.add_argument(
        '-k', '--key', default='cam/frames',
        help='HDF5 key for pupil frames'
    )
    parser.add_argument(
        '-n', '--number-of-cores', default=None,
        help='Number of cores to use for parallel processing'
    )
    parser.add_argument(
        '-o', '--output', default=None,
        help='Output HDF5 file for data'
    )
    parser.add_argument(
        '-p', '--plot', default=None,
        help='Output plot of pupil diameter to file'
    )
    parser.add_argument(
        '-t', '--threshold', default=128,
        help='Threshold to create binary image'
    )
    opts = parser.parse_args()
    bin_size = int(opts.bin_size)
    threshold = int(opts.threshold)

    if opts.number_of_cores:
        n_cores = int(opts.number_of_cores)
    else:
        n_cores = None

    # Create DataFrame
    time_limit = 300000
    ts = np.arange(0, time_limit, bin_size) + bin_size   # Don't start from 0
    
    nbins = len(ts)
    global df
    df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            zip(['base'] * nbins, ts) + zip(['ctrl'] * nbins, ts) + zip(['stim'] * nbins, ts),
            names=['epoch', 'time']
        ),
        columns=pd.MultiIndex(
            levels=[[], [], [], []],
            labels=[[], [], [], []],
            names=['subject', 'plane', 'order', 'feature']
        )
    )

    # Process file(s)
    if os.path.isdir(opts.data):
        files = glob.glob(os.path.join(opts.data, '*.h5'))
    elif os.path.isfile(opts.data):
        files = [opts.data]
    else:
        raise IOError('Invalid input "{}"'.format(opts.data))
    [manage_data(file, n_cores=n_cores, threshold=threshold, key=opts.key) for file in files]

    # Save DataFrame
    if opts.output:
        outfile = opts.output
    else:
        outfile = 'pupils.h5'

    df = df.sort_index(axis=1)
    with pd.HDFStore(outfile) as hf:
        hf['behav'] = df
        hf.get_storer('behav').attrs['threshold'] = threshold
        hf.get_storer('behav').attrs['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print('All done')


if __name__ == '__main__':
    main()
