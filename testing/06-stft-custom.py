#--------------------------------------------------
# Process STFT data as it's being parsed
# https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#stft
#--------------------------------------------------

import sys
import numpy as np
import librosa
import scipy

import os
import glob
import subprocess


# np.set_printoptions(threshold=np.nan)

MAX_MEM_BLOCK = 2**8 * 2**10

n_fft=2048
win_length = n_fft
hop_length = int(win_length // 4)
# hop_length = 64
window = 'hann'
dtype = np.complex64
dtype_size = dtype(0).itemsize # 8 bytes
pad_mode='reflect'

#--------------------------------------------------

if len(sys.argv) == 3:
    source_path = sys.argv[1]
    samples_folder = sys.argv[2]
else:
    source_path = './source-256.mp3'
    samples_folder = './06-stft-custom'

#--------------------------------------------------

print('Load Source')

if not os.path.exists(source_path):
    print('Missing source file')
    sys.exit()

source_series, source_rate = librosa.load(source_path)

source_time_total = (float(len(source_series)) / source_rate)

# source_data = abs(librosa.stft(source_series, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, dtype=dtype, pad_mode=pad_mode))

print('  {} ({})'.format(source_path, source_time_total))

#--------------------------------------------------

print('Load Samples')

samples = []

if not os.path.exists(samples_folder):
    print('Missing samples folder')
    sys.exit()

files = glob.glob(samples_folder + '/*')
for sample_path in files:

    sample_series, sample_rate = librosa.load(sample_path)

    sample_data = abs(librosa.stft(sample_series, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, dtype=dtype, pad_mode=pad_mode))

    sample_height = sample_data.shape[0]
    sample_length = sample_data.shape[1]

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
    sample_start += 2 # The first few frames seem to get modified, perhaps down to compression?

    sample_length = (sample_length - sample_start)

    samples.append([
            sample_start,
            sample_length,
            sample_data
        ])

    print('  {} ({}/{})'.format(sample_path, sample_start, sample_length))

#--------------------------------------------------
# Get Window

print('Get Window')

fft_window = scipy.signal.get_window(window, win_length, fftbins=True)

#--------------------------------------------------
# Pad the window out to n_fft size... Wrapper for
# np.pad to automatically centre an array prior to padding.

print('Pad Window')

axis = -1

n = fft_window.shape[axis]

lpad = int((n_fft - n) // 2)

lengths = [(0, 0)] * fft_window.ndim
lengths[axis] = (lpad, int(n_fft - n - lpad))

if lpad < 0:
    raise ParameterError(('Target size ({:d}) must be at least input size ({:d})').format(n_fft, n))

fft_window = np.pad(fft_window, lengths, mode='constant')

#--------------------------------------------------
# Reshape so that the window can be broadcast

print('Reshape Window')

fft_window = fft_window.reshape((-1, 1))

#--------------------------------------------------
# Pad the time series so that frames are centred

print('Pad time series')

source_series = np.pad(source_series, int(n_fft // 2), mode=pad_mode)

#--------------------------------------------------
# Window the time series.

print('Window time series')

# Compute the number of frames that will fit. The end may get truncated.
n_frames = 1 + int((len(source_series) - n_fft) / hop_length) # Where n_fft = frame_length

# Vertical stride is one sample
# Horizontal stride is `hop_length` samples
source_series_frames = np.lib.stride_tricks.as_strided(source_series, shape=(n_fft, n_frames), strides=(source_series.itemsize, hop_length * source_series.itemsize))
source_series_frame_count = source_series_frames.shape[1]

source_series_hz_count = int(1 + n_fft // 2) # 1025 (Hz buckets)

#--------------------------------------------------
# how many columns can we fit within MAX_MEM_BLOCK?

print('Columns')

n_columns = int(MAX_MEM_BLOCK / (source_series_hz_count * dtype_size))

#--------------------------------------------------
# Processing

print('Processing')

hz_diff_match = 0.005
hz_match_min = int(source_series_hz_count * 0.70) # i.e. "x% of 1025"

matching = {}
match_count = 0
matches = []

for block_start in range(0, source_series_frame_count, n_columns): # Time in 31 blocks

    block_end = min(block_start + n_columns, source_series_frame_count)

    set_data = abs((scipy.fftpack.fft(fft_window * source_series_frames[:, block_start:block_end], axis=0)).astype(dtype))

    print('  {} to {}'.format(block_start, block_end))

    x = 0
    x_max = (block_end - block_start)
    while x < x_max:

        matching_complete = []
        for matching_id in list(matching):

            sample_id = matching[matching_id][0]
            sample_x = (matching[matching_id][1] + 1)

            if sample_id in matching_complete:
                # print('    Match {}/{}: Duplicate Complete at {}'.format(sample_id, matching_id, sample_x))
                del matching[matching_id]
                continue;

            hz_matched = 0
            for y in range(0, source_series_hz_count):
                diff = set_data[y][x] - samples[sample_id][2][y][sample_x]
                if diff < 0:
                    diff = 0 - diff
                if diff < hz_diff_match:
                    hz_matched += 1

            if hz_matched > hz_match_min:
                if sample_x >= samples[sample_id][1]:
                    print('    Match {}/{}: Complete at {}'.format(sample_id, matching_id, sample_x))
                    del matching[matching_id]
                    matches.append([sample_id, ((float(x + block_start - samples[sample_id][1]) * hop_length) / source_rate)])
                    matching_complete.append(sample_id)
                else:
                    # print('    Match {}/{}: Update to {} via {}'.format(sample_id, matching_id, sample_x, hz_matched))
                    matching[matching_id][1] = sample_x
            else:
                print('    Match {}/{}: Failed at {} of {}'.format(sample_id, matching_id, sample_x, samples[sample_id][1]))
                del matching[matching_id]

        for sample_id, sample_info in enumerate(samples):

            sample_start = sample_info[0]

            hz_matched = 0
            for y in range(0, source_series_hz_count):
                diff = set_data[y][x] - sample_info[2][y][sample_start]
                if diff < 0:
                    diff = 0 - diff
                if diff < hz_diff_match:
                    hz_matched += 1

            if hz_matched > hz_match_min:
                match_count += 1
                # print('   Start Match {}'.format(match_count))
                matching[match_count] = [
                        sample_id,
                        sample_start
                    ]

        x += 1

    # print('{} - {}'.format(block_start, block_end))
    # for x in range(0, source_series_hz_count):
    #     for y in range(0, (block_end - block_start)):
    #         a = (set_data[x][y])
    #         b = (source_data[x][block_start + y])
    #         if a != b:
    #             print(' {} x {} ... {} != {}'.format(x, y, a, b))

#--------------------------------------------------

print('')
print('Matches')
print(matches)



