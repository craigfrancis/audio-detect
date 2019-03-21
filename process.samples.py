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

        sample_series = pcm_data(sample_path, sample_rate)

        if sample_max_length < sample_series.shape[0]:
            sample_max_length = sample_series.shape[0]

        samples.append([
                sample_path,
                sample_series,
            ])

#--------------------------------------------------

print('')
print(sample_max_length)
print('')

for sample_id, sample_info in enumerate(samples):

    #--------------------------------------------------
    # Config

    sample_path = sample_info[0]
    sample_series = sample_info[1]

    sample_path_split = os.path.split(sample_path)
    sample_ext_split = os.path.splitext(sample_path_split[1])
    sample_image_path = sample_path_split[0] + '/img/' + sample_ext_split[0] + '.png'

    #--------------------------------------------------
    # All samples the same length

    series_length = sample_series.shape[0]
    if sample_max_length > series_length:
        empty_series = np.full((sample_max_length - series_length), 0)
        sample_series = np.concatenate((sample_series, empty_series), axis=0)

    #--------------------------------------------------
    # Harmonic and percussive components

    y_harm, y_perc = librosa.effects.hpss(sample_series)

    #--------------------------------------------------
    # STFT data

    sample_frames, fft_window, n_columns = stft_raw(sample_series, sample_rate, win_length, hop_length, hz_count, dtype)

    # Pre-allocate the STFT matrix
    sample_data = np.empty((int(1 + n_fft // 2), sample_frames.shape[1]), dtype=dtype, order='F')

    for bl_s in range(0, sample_data.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, sample_data.shape[1])
        sample_data[:, bl_s:bl_t] = scipy.fftpack.fft(fft_window * sample_frames[:, bl_s:bl_t], axis=0)[:sample_data.shape[0]]

    sample_data = abs(sample_data)

    sample_height = sample_data.shape[0]
    sample_length = sample_data.shape[1]

    #--------------------------------------------------
    # Start

    x = 0
    sample_start = 0
    while x < sample_length:
        total = 0
        for y in range(0, sample_height):
            total += sample_data[y][x]
        if total >= 1:
            sample_start = x
            break
        x += 1
    sample_start += sample_crop_start # The first few frames seem to get modified, perhaps down to compression?
    sample_start = ((float(sample_start) * hop_length) / sample_rate);

    #--------------------------------------------------
    # Plot

    plt.figure(figsize=(5, 6))

    plt.subplot(2, 1, 1)
    librosa.display.waveplot(y_harm, sr=sample_rate, alpha=0.25)
    librosa.display.waveplot(y_perc, sr=sample_rate, color='r', alpha=0.5)
    plt.axvline(x=sample_start)
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    librosa.display.specshow(sample_data, sr=sample_rate, x_axis='time', y_axis='log', cmap='Reds')
    plt.axvline(x=sample_start)
    plt.tight_layout()

    plt.savefig(sample_image_path)

    #--------------------------------------------------
    # Done

    print('  {} - {}'.format(sample_ext_split[0], series_length))

#--------------------------------------------------

print('')
print('Done')
print('')
