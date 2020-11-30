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
import re

#--------------------------------------------------

filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'process.source.py');

exec(open(filename).read())

#--------------------------------------------------

samples_folder = 'samples'

if len(sys.argv) == 2:
    samples_folder = sys.argv[1]

if not os.path.exists(samples_folder):
    print('Missing samples folder: ' + samples_folder)
    sys.exit()

#--------------------------------------------------

samples = []
series_max_length = 0

if os.path.isdir(samples_folder):
    files = sorted(glob.glob(os.path.join(samples_folder, '*')))
else:
    files = [samples_folder]

for sample_path in files:
    if os.path.isfile(sample_path):

        series_data = pcm_data(sample_path, sample_rate)

        if series_max_length < series_data.shape[0]:
            series_max_length = series_data.shape[0]

        samples.append([
                sample_path,
                series_data,
            ])

#--------------------------------------------------

print('')
print(series_max_length)
print('')

for sample_id, sample_info in enumerate(samples):

    #--------------------------------------------------
    # Config

    sample_path = sample_info[0]
    sample_path_split = os.path.split(sample_path)
    sample_ext_split = os.path.splitext(sample_path_split[1])

    series_data = sample_info[1]

    #--------------------------------------------------
    # Original frame length

    stft_frames, fft_window, n_columns = stft_raw(series_data, sample_rate, win_length, hop_length, hz_count, dtype)

    stft_length_source = stft_frames.shape[1]

    #--------------------------------------------------
    # All samples the same length

    series_length = series_data.shape[0]
    if series_max_length > series_length:
        series_padding = np.full((series_max_length - series_length), 0)
        series_data = np.concatenate((series_data, series_padding), axis=0)

    #--------------------------------------------------
    # Harmonic and percussive components

    series_harm, series_perc = librosa.effects.hpss(series_data)

    #--------------------------------------------------
    # STFT data

    stft_frames, fft_window, n_columns = stft_raw(series_data, sample_rate, win_length, hop_length, hz_count, dtype)

    # Pre-allocate the STFT matrix
    stft_data = np.empty((int(1 + n_fft // 2), stft_frames.shape[1]), dtype=dtype, order='F')

    for bl_s in range(0, stft_data.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_data.shape[1])
        stft_data[:, bl_s:bl_t] = scipy.fft.fft(fft_window * stft_frames[:, bl_s:bl_t], axis=0)[:stft_data.shape[0]]

    stft_data = abs(stft_data)

    stft_height = stft_data.shape[0]
    stft_length_padded = stft_data.shape[1]

    #--------------------------------------------------
    # Start

    x = 0
    stft_crop_start = 0
    while x < stft_length_padded:
        total = 0
        for y in range(0, stft_height):
            total += stft_data[y][x]
        if total >= 1:
            stft_crop_start = x
            break
        x += 1
    stft_crop_start += sample_crop_start
    stft_crop_end = (stft_length_source - sample_crop_end)

    stft_crop_start_time = ((float(stft_crop_start) * hop_length) / sample_rate)
    stft_crop_end_time = ((float(stft_crop_end) * hop_length) / sample_rate)

    #--------------------------------------------------
    # Plot

    plt.figure(figsize=(5, 6))

    plt.subplot(2, 1, 1)
    librosa.display.waveplot(series_harm, sr=sample_rate, alpha=0.25)
    librosa.display.waveplot(series_perc, sr=sample_rate, color='r', alpha=0.5)
    plt.axvline(x=stft_crop_start_time)
    plt.axvline(x=stft_crop_end_time)
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    librosa.display.specshow(stft_data, sr=sample_rate, x_axis='time', y_axis='log', cmap='Reds')
    plt.axvline(x=stft_crop_start_time)
    plt.axvline(x=stft_crop_end_time)
    plt.tight_layout()

    plt.savefig(os.path.join(sample_path_split[0], 'img', sample_ext_split[0] + '.png'))

    #--------------------------------------------------
    # Details

    details = {}
    detail_path = os.path.join(sample_path_split[0], 'info', sample_ext_split[0] + '.txt');

    if os.path.exists(detail_path):
        p = re.compile('([^:]+): *(.*)')
        f = open(detail_path, 'r')
        for line in f:
            m = p.match(line)
            if m:
                details[m.group(1)] = m.group(2)

    details['crop_start'] = str(stft_crop_start)
    details['crop_end'] = str(stft_crop_end)
    details['length_series'] = str(series_length)

    f = open(detail_path, 'w')
    for field in sorted(iter(details.keys())):
        f.write(field + ': ' + details[field] + '\n')

    #--------------------------------------------------
    # Done

    print('  {} ({}/{}) - {}'.format(sample_path, stft_crop_start, stft_crop_end, series_length))

#--------------------------------------------------

print('')
print('Done')
print('')
