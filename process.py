#--------------------------------------------------
# Process STFT data as it's being parsed
# https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#stft
#--------------------------------------------------

import sys
import os
import subprocess
import numpy as np
import scipy
import scipy.signal
import datetime

exec(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'process.source.py')).read())

#--------------------------------------------------

config = {

        'ffmpeg_path':        'ffmpeg', # On Windows, you will probably need this to be \path\to\ffmpeg.exe

        'source_path':        os.path.join('testing', 'source-64.mp3'),
        'source_frame_start': 0,    # (x * sample_rate) / hop_length)
        'source_frame_end':   None, # (x * sample_rate) / hop_length)

        'matching_samples':   os.path.join('testing', '06-stft-custom', 'sample-1a.mp3'),
        'matching_min_score': 0.15,
        'matching_skip':      0,    # Jump forward X seconds after a match.
        'matching_ignore':    0,    # Ignore additional matches X seconds after the last one.

        'output_title':       None, # Set a title to create ".meta" file, and "X-chapters.mp3"

    }

if len(sys.argv) >= 2:
    config_path = sys.argv[1]
    if config_path != None:
        exec(open(config_path).read())

if len(sys.argv) >= 3:
    config['source_path'] = sys.argv[2]

print('Config')
print('  Hz Min Score: {}'.format(config['matching_min_score']))

#--------------------------------------------------

start_time = datetime.datetime.now()

#--------------------------------------------------

print('Load Source')

if not os.path.exists(config['source_path']):
    print('Missing source file')
    sys.exit()

source_series = pcm_data(config['source_path'], sample_rate)

source_time_total = (float(len(source_series)) / sample_rate)

print('  {} ({} & {})'.format(config['source_path'], source_time_total, sample_rate))

#--------------------------------------------------

print('Load Samples')

samples = []

if not os.path.exists(config['matching_samples']):
    print('Missing samples folder: ' + config['matching_samples'])
    sys.exit()

if os.path.isdir(config['matching_samples']):
    files = [];
    for path in os.listdir(config['matching_samples']):
        path = os.path.join(config['matching_samples'], path)
        if os.path.isfile(path) and not os.path.basename(path).startswith('.'):
            files.append(path)
    files = sorted(files)
else:
    files = [config['matching_samples']]

for sample_path in files:
    if os.path.isfile(sample_path):

        sample_series = pcm_data(sample_path, sample_rate)

        sample_frames, fft_window, n_columns = stft_raw(sample_series, sample_rate, win_length, hop_length, hz_count, dtype)

        # Pre-allocate the STFT matrix
        sample_data = np.empty((int(1 + n_fft // 2), sample_frames.shape[1]), dtype=dtype, order='F')

        for bl_s in range(0, sample_data.shape[1], n_columns):
            bl_t = min(bl_s + n_columns, sample_data.shape[1])
            sample_data[:, bl_s:bl_t] = scipy.fft.fft(fft_window * sample_frames[:, bl_s:bl_t], axis=0)[:sample_data.shape[0]]

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
        sample_start += sample_crop_start # The first few frames seem to get modified, perhaps due to compression?
        sample_end = (sample_length - sample_crop_end)

        samples.append([
                sample_start,
                sample_end,
                os.path.basename(sample_path),
                sample_data
            ])

        print('  {} ({}/{})'.format(sample_path, sample_start, sample_end))

#--------------------------------------------------
# Processing

print('Processing')

source_frames, fft_window, n_columns = stft_raw(source_series, sample_rate, win_length, hop_length, hz_count, dtype)

if config['source_frame_end'] == None:
   config['source_frame_end'] = source_frames.shape[1]

print('    From {} to {}'.format(config['source_frame_start'], config['source_frame_end']))
print('    From {} to {}'.format(((float(config['source_frame_start']) * hop_length) / sample_rate), ((float(config['source_frame_end']) * hop_length) / sample_rate)))

matching = {}
match_count = 0
match_last_time = None
match_last_ignored = False
match_skipping = 0
matches = []

results_end = {}
results_dupe = {}
for sample_id, sample_info in enumerate(samples):
    results_end[sample_id] = {}
    results_dupe[sample_id] = {}
    for k in range(0, (sample_info[1] + 1)):
        results_end[sample_id][k] = 0
        results_dupe[sample_id][k] = 0

for block_start in range(config['source_frame_start'], config['source_frame_end'], n_columns): # Time in 31 blocks

    block_end = min(block_start + n_columns, config['source_frame_end'])

    set_data = abs((scipy.fft.fft(fft_window * source_frames[:, block_start:block_end], axis=0)).astype(dtype))

    print('  {} to {} - {}'.format(block_start, block_end, str(datetime.timedelta(seconds=((float(block_start) * hop_length) / sample_rate)))))

    x = 0
    x_max = (block_end - block_start)
    while x < x_max:

        if match_skipping > 0:
            if x == 0:
                print('    Skipping {}'.format(match_skipping))
            match_skipping -= 1
            x += 1
            continue

        matching_complete = []
        for matching_id in list(matching): # Continue to check matches (i.e. have already started)

            sample_id = matching[matching_id][0]
            sample_x = (matching[matching_id][1] + 1)

            if sample_id in matching_complete:
                continue

# TEST-2... this is the main test (done after the first frame has been matched with TEST-1)

              ###
              # While this does not work, maybe we could try something like this?
              #
              #     match_min_score = (0 - config['matching_min_score']);
              #
              #     hz_score = (set_data[0:hz_count,x] - samples[sample_id][3][0:hz_count,sample_x])
              #     hz_score = (hz_score < match_min_score).sum()
              #
              #     if hz_score < 5:
              #
              ###
              # Correlation might work better, but I've no idea how to use it.
              #   np.correlate(set_data[0:hz_count,x], sample_info[3][0:hz_count,sample_start])[0]
              ###

            # Return a list of Hz buckets for this frame (set_data[0-1025][x]),
            # This is where `hz_score` starts as a simple array, using a column of results at time position `x`.
            # Subtract them all from the equivalent Hz bucket from sample_start (frame 0, ish)
            # Convert to positive values (abs),
            # Calculate the average variation, as a float (total/count).

            hz_score = abs(set_data[0:hz_count,x] - samples[sample_id][3][0:hz_count,sample_x])
            hz_score = sum(hz_score)/float(len(hz_score))

            if hz_score < config['matching_min_score']:

                if sample_x >= samples[sample_id][1]:

                    match_start_time = ((float(x + block_start - samples[sample_id][1]) * hop_length) / sample_rate)

                    print('    Match {}/{}: Complete at {} @ {}'.format(matching_id, sample_id, sample_x, match_start_time))

                    results_end[sample_id][sample_x] += 1

                    if (config['matching_skip']) or (match_last_time == None) or ((match_start_time - match_last_time) > config['matching_ignore']):
                        match_last_ignored = False
                    else:
                        match_last_ignored = True

                    matches.append([sample_id, match_start_time, match_last_ignored])
                    match_last_time = match_start_time

                    if config['matching_skip']:
                        match_skipping = ((config['matching_skip'] * sample_rate) / hop_length)
                        print('    Skipping {}'.format(match_skipping))
                        matching = {}
                        break # No more 'matching' entires
                    else:
                        del matching[matching_id]
                        matching_complete.append(sample_id)

                else:

                    print('    Match {}/{}: Update to {} ({} < {})'.format(matching_id, sample_id, sample_x, hz_score, config['matching_min_score']))
                    matching[matching_id][1] = sample_x

            elif matching[matching_id][2] < sample_warn_allowance and sample_x > 10:

                print('    Match {}/{}: Warned at {} of {} ({} > {})'.format(matching_id, sample_id, sample_x, samples[sample_id][1], hz_score, config['matching_min_score']))
                matching[matching_id][2] += 1

            else:

                print('    Match {}/{}: Failed at {} of {} ({} > {})'.format(matching_id, sample_id, sample_x, samples[sample_id][1], hz_score, config['matching_min_score']))
                results_end[sample_id][sample_x] += 1
                del matching[matching_id]

        if match_skipping > 0:
            continue

        for matching_sample_id in matching_complete:
            for matching_id in list(matching):
                if match_any_sample or matching[matching_id][0] == matching_sample_id:
                    sample_id = matching[matching_id][0]
                    sample_x = matching[matching_id][1]
                    print('    Match {}/{}: Duplicate Complete at {}'.format(matching_id, sample_id, sample_x))
                    results_dupe[sample_id][sample_x] += 1
                    del matching[matching_id] # Cannot be done in the first loop (next to continue), as the order in a dictionary is undefined, so you could have a match that started later, getting tested first.

        for sample_id, sample_info in enumerate(samples): # For each sample, see if the first frame (after sample_crop_start), matches well enough to keep checking (that part is done above).

            sample_start = sample_info[0]

# TEST-1

            hz_score = abs(set_data[0:hz_count,x] - sample_info[3][0:hz_count,sample_start])
            hz_score = sum(hz_score)/float(len(hz_score))

            if hz_score < config['matching_min_score']:

                match_count += 1
                print('    Match {}: Start for sample {} at {} ({} < {})'.format(match_count, sample_id, (x + block_start), hz_score, config['matching_min_score']))
                matching[match_count] = [
                        sample_id,
                        sample_start,
                        0, # Warnings
                    ]

        x += 1

#--------------------------------------------------

print('')
print('Matches')
for match in matches:
    print(' {} = {} @ {}{}'.format(samples[match[0]][2], str(datetime.timedelta(seconds=match[1])), match[1], (' - Ignored' if match[2] else '')))

if config['output_title'] != None:

    source_path_split = os.path.splitext(config['source_path'])
    meta_path = source_path_split[0] + '.meta'
    results_path = source_path_split[0] + '.results'
    chapter_path = source_path_split[0] + '-chapters' + source_path_split[1]

    f = open(results_path, 'w')
    for sample_id, sample_info in enumerate(samples):
        for k in range(0, (sample_info[1] + 1)):
            f.write('  ' + str(sample_id) + ' | ' + str(sample_info[2]) + ' | ' + str(k))
            if k == sample_info[1]:
                f.write(' | L: ')
            else:
                f.write(' | P: ')
            if results_end[sample_id][k] > 0 or results_dupe[sample_id][k] > 0:
                f.write(' ' + str(results_end[sample_id][k]))
                if results_dupe[sample_id][k] > 0:
                    f.write(' (+' + str(results_dupe[sample_id][k]) + ')')
            f.write('\n')
        f.write('\n')
    f.close()

    f = open(meta_path, 'w')
    f.write(';FFMETADATA1\n')
    f.write('title=' + config['output_title'] + '\n')
    f.write('\n')
    k = 0
    last_time = 0
    last_sample = 'N/A'
    for match in matches:
        end_time = int(round(match[1]))
        if match[2] == True: # Not ignored
            f.write('#ignored=' + str(end_time * 1000) + ' (' + str(datetime.timedelta(seconds=end_time)) + ')\n')
            f.write('\n')
        else:
            k += 1
            f.write('[CHAPTER]\n')
            f.write('TIMEBASE=1/1000\n')
            f.write('START=' + str(last_time * 1000) + '\n')
            f.write('END=' + str(end_time * 1000) + '\n')
            f.write('title=Chapter ' + str(k) + '\n')
            f.write('#human-start=' + str(datetime.timedelta(seconds=last_time)) + '\n')
            f.write('#human-end=' + str(datetime.timedelta(seconds=end_time)) + '\n')
            f.write('#sample=' + str(last_sample) + '\n')
            f.write('\n')
            last_time = end_time
            last_sample = samples[match[0]][2]
    if last_time > 0:
        k += 1
        end_time = int(round((float(config['source_frame_end']) * hop_length) / sample_rate))
        f.write('[CHAPTER]\n')
        f.write('TIMEBASE=1/1000\n')
        f.write('START=' + str(last_time * 1000) + '\n')
        f.write('END=' + str(end_time * 1000) + '\n')
        f.write('title=Chapter ' + str(k) + '\n')
        f.write('#human-start=' + str(datetime.timedelta(seconds=last_time)) + '\n')
        f.write('#human-end=' + str(datetime.timedelta(seconds=end_time)) + '\n')
        f.write('#sample=' + str(last_sample) + '\n')
        f.write('\n')
    f.close()

    devnull = open(os.devnull)
    proc = subprocess.Popen([config['ffmpeg_path'], '-i', config['source_path'], '-i', meta_path, '-map_metadata', '1', '-codec', 'copy', '-y', chapter_path], stdin=devnull, stdout=devnull, stderr=devnull)
    devnull.close()

#--------------------------------------------------

print('')
print(datetime.datetime.now() - start_time)
