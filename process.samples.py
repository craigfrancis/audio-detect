#--------------------------------------------------

import sys
import os
import glob
import subprocess
import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt
import librosa.display

execfile(os.path.dirname(os.path.realpath(__file__)) + '/process.source.py')

#--------------------------------------------------

samples_folder = './samples'

if len(sys.argv) == 2:
    samples_folder = sys.argv[1]

if not os.path.exists(samples_folder):
    print('Missing samples folder')
    sys.exit()

#--------------------------------------------------

samples = []
sample_max_length = 0

if os.path.isdir(samples_folder):
    files = sorted(glob.glob(samples_folder + '/*'))
else:
    files = [samples_folder]

for sample_path in files:
    if os.path.isfile(sample_path):

        sample_path_split = os.path.split(sample_path)
        sample_ext_split = os.path.splitext(sample_path_split[1])
        sample_image_path = sample_path_split[0] + '/img/' + sample_ext_split[0] + '.png'

        sample_series = pcm_data(sample_path, sample_rate)

        sample_frames, fft_window, n_columns = stft_raw(sample_series, sample_rate, win_length, hop_length, hz_count, dtype)

        if sample_max_length < sample_frames.shape[1]:
            sample_max_length = sample_frames.shape[1]

        samples.append([
                sample_image_path,
                sample_frames,
                fft_window,
                n_columns
            ])

#--------------------------------------------------

print('')
print(sample_max_length)
print('')

for sample_id, sample_info in enumerate(samples):

    sample_image_path = sample_info[0]
    sample_frames = sample_info[1]
    fft_window = sample_info[2]
    n_columns = sample_info[3]

    sample_length = sample_frames.shape[1]

    if sample_max_length > sample_length:
        empty_frames = np.full((n_fft, (sample_max_length - sample_length)), 0)
        sample_frames = np.concatenate((sample_frames, empty_frames), axis=1)

    # Pre-allocate the STFT matrix
    sample_data = np.empty((int(1 + n_fft // 2), sample_max_length), dtype=dtype, order='F')

    for bl_s in range(0, sample_max_length, n_columns):
        bl_t = min(bl_s + n_columns, sample_max_length)
        sample_data[:, bl_s:bl_t] = scipy.fftpack.fft(fft_window * sample_frames[:, bl_s:bl_t], axis=0)[:sample_data.shape[0]]

    sample_data = abs(sample_data)

    plt.figure(figsize=(5, 4))
    librosa.display.specshow(sample_data, sr=sample_rate, x_axis='time', y_axis='log', cmap='Reds')
    plt.savefig(sample_image_path)

    print('  {} - {}'.format(sample_image_path, sample_length))

#--------------------------------------------------

print('')
print('Done')
print('')
