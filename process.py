#--------------------------------------------------
# Process STFT data as it's being parsed
# https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#stft
#--------------------------------------------------

import sys
import os
import glob
import subprocess
import numpy as np
import scipy
import scipy.signal
import datetime

# np.set_printoptions(threshold=np.nan)

dtype = np.complex64
n_fft=2048
hz_count = int(1 + n_fft // 2) # 1025 (Hz buckets)
win_length = n_fft
hop_length = int(win_length // 4)
# hop_length = 64
sample_rate = 22050

match_any_sample = True

#--------------------------------------------------

def pcm_data(path, sample_rate):

    devnull = open(os.devnull)
    proc = subprocess.Popen(['ffmpeg', '-i', path, '-f', 's16le', '-ac', '1', '-ar', str(sample_rate), '-'], stdout=subprocess.PIPE, stderr=devnull)
    devnull.close()

    scale = 1./float(1 << ((8 * 2) - 1))
    y = scale * np.frombuffer(proc.stdout.read(), '<i2').astype(np.float32)

    return y

#--------------------------------------------------

def stft_raw(series, sample_rate, win_length, hop_length, hz_count, dtype):

    #--------------------------------------------------
    # Config

    window = 'hann'
    pad_mode='reflect'

    #--------------------------------------------------
    # Get Window

    fft_window = scipy.signal.get_window(window, win_length, fftbins=True)

    #--------------------------------------------------
    # Pad the window out to n_fft size... Wrapper for
    # np.pad to automatically centre an array prior to padding.

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

    fft_window = fft_window.reshape((-1, 1))

    #--------------------------------------------------
    # Pad the time series so that frames are centred

    series = np.pad(series, int(n_fft // 2), mode=pad_mode)

    #--------------------------------------------------
    # Window the time series.

        # Compute the number of frames that will fit. The end may get truncated.
    n_frames = 1 + int((len(series) - n_fft) / hop_length) # Where n_fft = frame_length

        # Vertical stride is one sample
        # Horizontal stride is `hop_length` samples
    series_frames = np.lib.stride_tricks.as_strided(series, shape=(n_fft, n_frames), strides=(series.itemsize, hop_length * series.itemsize))

    #--------------------------------------------------
    # how many columns can we fit within MAX_MEM_BLOCK

    MAX_MEM_BLOCK = 2**8 * 2**10
    n_columns = int(MAX_MEM_BLOCK / (hz_count * (dtype(0).itemsize)))

    #--------------------------------------------------
    # Return

    return (series_frames, fft_window, n_columns)

#--------------------------------------------------

source_path = './testing/source-64.mp3'
samples_folder = './testing/06-stft-custom/sample-1a.mp3'
hz_min_score = 0.15
meta_title = None
source_frame_start = 0
source_frame_end = None

if len(sys.argv) >= 3:
    source_path = sys.argv[1]
    samples_folder = sys.argv[2]

if len(sys.argv) >= 4:
    hz_min_score = float(sys.argv[3])

if len(sys.argv) >= 5:
    meta_title = str(sys.argv[4])

if len(sys.argv) >= 6:
    source_frame_start = ((int(sys.argv[5]) * sample_rate) / hop_length)

if len(sys.argv) >= 7:
    source_frame_end = ((int(sys.argv[6]) * sample_rate) / hop_length)

print('Config')
print('  Hz Min Score: {}'.format(hz_min_score))

#--------------------------------------------------

print('Load Source')

if not os.path.exists(source_path):
    print('Missing source file')
    sys.exit()

source_series = pcm_data(source_path, sample_rate)

source_time_total = (float(len(source_series)) / sample_rate)

print('  {} ({} & {})'.format(source_path, source_time_total, sample_rate))

#--------------------------------------------------

print('Load Samples')

samples = []

if not os.path.exists(samples_folder):
    print('Missing samples folder')
    sys.exit()

if os.path.isdir(samples_folder):
    files = sorted(glob.glob(samples_folder + '/*'))
else:
    files = [samples_folder]

for sample_path in files:

    sample_series = pcm_data(sample_path, sample_rate)

    sample_frames, fft_window, n_columns = stft_raw(sample_series, sample_rate, win_length, hop_length, hz_count, dtype)

    # Pre-allocate the STFT matrix
    sample_data = np.empty((int(1 + n_fft // 2), sample_frames.shape[1]), dtype=dtype, order='F')

    for bl_s in range(0, sample_data.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, sample_data.shape[1])
        sample_data[:, bl_s:bl_t] = scipy.fftpack.fft(fft_window * sample_frames[:, bl_s:bl_t], axis=0)[:sample_data.shape[0]]

    sample_data = abs(sample_data)

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
    sample_start += 5 # The first few frames seem to get modified, perhaps down to compression?

    sample_length = (sample_length - sample_start - 2)

    samples.append([
            sample_start,
            sample_length,
            os.path.basename(sample_path),
            sample_data
        ])

    print('  {} ({}/{})'.format(sample_path, sample_start, sample_length))

#--------------------------------------------------
# Processing

print('Processing')

source_frames, fft_window, n_columns = stft_raw(source_series, sample_rate, win_length, hop_length, hz_count, dtype)

if source_frame_end == None:
   source_frame_end = source_frames.shape[1]

print('    From {} to {}'.format(source_frame_start, source_frame_end))
print('    From {} to {}'.format(((float(source_frame_start) * hop_length) / sample_rate), ((float(source_frame_end) * hop_length) / sample_rate)))

matching = {}
match_count = 0
matches = []

results = {}
for sample_id, sample_info in enumerate(samples):
    results[sample_id] = {}
    for k in range(0, (sample_info[1] + 1)):
        results[sample_id][k] = 0

for block_start in range(source_frame_start, source_frame_end, n_columns): # Time in 31 blocks

    block_end = min(block_start + n_columns, source_frame_end)

    set_data = abs((scipy.fftpack.fft(fft_window * source_frames[:, block_start:block_end], axis=0)).astype(dtype))

    print('  {} to {} @ {}'.format(block_start, block_end, ((float(block_start) * hop_length) / sample_rate)))

    x = 0
    x_max = (block_end - block_start)
    while x < x_max:

        matching_complete = []
        for matching_id in list(matching):

            sample_id = matching[matching_id][0]
            sample_x = (matching[matching_id][1] + 1)

            if sample_id in matching_complete:
                continue

            hz_score = abs(set_data[0:hz_count,x] - samples[sample_id][3][0:hz_count,sample_x])
            hz_score = sum(hz_score)/float(len(hz_score))

            if hz_score < hz_min_score:
                if sample_x >= samples[sample_id][1]:
                    match_start_time = ((float(x + block_start - samples[sample_id][1]) * hop_length) / sample_rate)
                    print('    Match {}/{}: Complete at {} @ {}'.format(matching_id, sample_id, sample_x, match_start_time))
                    results[sample_id][sample_x] += 1
                    del matching[matching_id]
                    matches.append([sample_id, match_start_time])
                    matching_complete.append(sample_id)
                else:
                    print('    Match {}/{}: Update to {} ({} < {})'.format(matching_id, sample_id, sample_x, hz_score, hz_min_score))
                    matching[matching_id][1] = sample_x
            else:
                print('    Match {}/{}: Failed at {} of {} ({} > {})'.format(matching_id, sample_id, sample_x, samples[sample_id][1], hz_score, hz_min_score))
                results[sample_id][sample_x] += 1
                del matching[matching_id]

        for sample_id in matching_complete:
            for matching_id in list(matching):
                if match_any_sample or matching[matching_id][0] == sample_id:
                    print('    Match {}/{}: Duplicate Complete at {}'.format(matching_id, sample_id, sample_x))
                    del matching[matching_id] # Cannot be done in the first loop (next to continue), as the order in a dictionary is undefined, so you could have a match that started later, getting tested first.

        for sample_id, sample_info in enumerate(samples):

            sample_start = sample_info[0]

            # Correlation might work better, but I've no idea how to use it.
            #   np.correlate(set_data[0:hz_count,x], sample_info[3][0:hz_count,sample_start])[0]

            # Return a list of Hz buckets for this frame (set_data[0-1025][x]),
            # Subtract them all from the equivalent Hz bucket from sample_start (frame 0, ish)
            # Convert to positive values (abs),
            # Calculate the average variation, as a float (total/count).

            hz_score = abs(set_data[0:hz_count,x] - sample_info[3][0:hz_count,sample_start])
            hz_score = sum(hz_score)/float(len(hz_score))

            if hz_score < hz_min_score:
                match_count += 1
                print('    Match {}: Start for sample {} at {} ({} < {})'.format(match_count, sample_id, (x + block_start), hz_score, hz_min_score))
                matching[match_count] = [
                        sample_id,
                        sample_start
                    ]

        x += 1

#--------------------------------------------------

print('')
print('Matches')
for match in matches:
    print(' {} = {} @ {}'.format(match[0], str(datetime.timedelta(seconds=match[1])), match[1]))

if meta_title != None:

    source_path_split = os.path.splitext(source_path)
    meta_path = source_path_split[0] + '.meta'
    results_path = source_path_split[0] + '.results'
    chapter_path = source_path_split[0] + '-chapters' + source_path_split[1]

    f = open(results_path, 'w')
    for sample_id, sample_info in enumerate(samples):
        f.write('  ' + str(sample_id) + ' / ' + str(sample_info[2]) + '\n')
        for k in range(0, (sample_info[1] + 1)):
            if results[sample_id][k] > 0:
                f.write('    ' + str(k) + ': ' + str(results[sample_id][k]) + '\n')
            else:
                f.write('    ' + str(k) + ':\n')

    f = open(meta_path, 'w')
    f.write(';FFMETADATA1\n')
    f.write('title=' + meta_title + '\n')
    f.write('\n')
    k = 0
    last = 0
    for match in matches:
        k += 1
        time = int(round(match[1]))
        f.write('[CHAPTER]\n')
        f.write('TIMEBASE=1/1000\n')
        f.write('START=' + str(last * 1000) + '\n')
        f.write('END=' + str(time * 1000) + '\n')
        f.write('title=Chapter ' + str(k) + '\n')
        f.write('#human-start=' + str(str(datetime.timedelta(seconds=last))) + '\n')
        f.write('#human-end=' + str(str(datetime.timedelta(seconds=time))) + '\n')
        f.write('#sample=' + str(samples[match[0]][2]) + '\n')
        f.write('\n')
        last = time
    if last > 0:
        k += 1
        time = int(round(match[1]))
        end = int(round((float(source_frame_end) * hop_length) / sample_rate))
        f.write('[CHAPTER]\n')
        f.write('TIMEBASE=1/1000\n')
        f.write('START=' + str(time * 1000) + '\n')
        f.write('END=' + str(end * 1000) + '\n')
        f.write('title=Chapter ' + str(k) + '\n')
        f.write('#human-start=' + str(str(datetime.timedelta(seconds=time))) + '\n')
        f.write('#human-end=' + str(str(datetime.timedelta(seconds=end))) + '\n')
        f.write('#sample=' + str(samples[match[0]][2]) + '\n')
        f.write('\n')
    f.close()

    devnull = open(os.devnull)
    proc = subprocess.Popen(['ffmpeg', '-i', source_path, '-i', meta_path, '-map_metadata', '1', '-codec', 'copy', '-y', chapter_path], stdin=devnull, stdout=devnull, stderr=devnull)
    devnull.close()
