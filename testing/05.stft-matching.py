#--------------------------------------------------
# Basic matching... kind of works, but very slow.
#
#  python 05.stft-matching.py 1a
#    FOUND @ 1.63989499419
#    FOUND @ 5.41600718434
#    FOUND @ 7.89471572423
#
#  python 05.stft-matching.py 2a
#    FOUND @ 1.60216289698
#    FOUND @ 5.37827508713
#    FOUND @ 7.85698362702
#
#--------------------------------------------------

import sys
import numpy as np
import librosa

import os
import glob
import subprocess

#--------------------------------------------------

if len(sys.argv) == 2:
    sample_id = sys.argv[1]
else:
    print('Missing Sample ID')
    sys.exit()

source_path = './source-256.mp3'
sample_path = './03.series-big/sample-' + sample_id + '.mp3'

#--------------------------------------------------

matches_path = './05.stft-matching'

if os.path.exists(matches_path):
    files = glob.glob(matches_path + '/*')
    for f in files:
        os.remove(f)
else:
    print('Missing matches folder')
    sys.exit()

#--------------------------------------------------

print('Load files')

source_series, source_rate = librosa.load(source_path)
sample_series, sample_rate = librosa.load(sample_path)

source_time_total = (float(len(source_series)) / source_rate)

#--------------------------------------------------

print('Parse data')

source_data = abs(librosa.stft(source_series, hop_length=64))
sample_data = abs(librosa.stft(sample_series, hop_length=64))

#--------------------------------------------------

print('Details')

source_height = source_data.shape[0]
sample_height = sample_data.shape[0]

source_length = source_data.shape[1]
sample_length = sample_data.shape[1]

source_timing = (float(source_time_total) / source_length)

print('')
print('  Height: {} and {}'.format(sample_height, source_height))
print('  Source Length: {} / {}'.format(source_length, (source_length * source_timing)))
print('  Sample Length: {} / {}'.format(sample_length, (sample_length * source_timing)))
print('')

#--------------------------------------------------

print('Sample start')

sample_start = 0

x = 0
while x < sample_length:
    total = 0
    for y in range(0, sample_height):
        total += sample_data[y][x]
    if total >= 1:
        sample_start = x
        break
    x += 1

sample_start += 10

print('')
print('  {} of {}'.format(sample_start, sample_length))
print('')

sample_length = (sample_length - sample_start)

#--------------------------------------------------

print('Matching requirements')

hz_diff_match = 0.005
hz_match_min = int(sample_height * 0.70) # i.e. "x% of 1025"

    #--------------------------------------------------
    # Comparing the sample to the source, by checking the 1025
    # frequency buckets for each frame.
    #
    # If the difference requirement was set to no more than 0.005,
    # then the total for each frame, for these three samples, got
    # between 718 and 895 matches.
    #
    # Which is why the min matches is set to 718/1025 = 0.700
    #
    # Changing the match requirements does the following:
    #
    #   0.0100 needs to use 0.701 (719 to 873) with 2594 resets, taking 1:27, +1 false positive
    #   0.0050 needs to use 0.700 (718 to 895) with  194 resets, taking 0:20, first false positive at 670.
    #   0.0010 needs to use 0.620 (640 to 889) with   55 resets, taking 0:15, first false positive at 593.
    #   0.0005 needs to use 0.509 (522 to 853) with   98 resets, taking 0:20, first false positive at 518.
    #
    #--------------------------------------------------

print('')
print('  Diff Match: {}'.format(hz_diff_match))
print('  Match Min: {}'.format(hz_match_min))
print('')

#--------------------------------------------------

print('Process series')
print('')

source_start = -1
sample_matching = sample_start
sample_matches_counts = []
reset_count = 0
match_count = 0
match_min = None
match_max = None

x = 0
while x < source_length:

    hz_matched = 0
    for y in range(0, sample_height):
        diff = source_data[y][x] - sample_data[y][sample_matching]
        if diff < 0:
            diff = 0 - diff
        if diff < hz_diff_match:
            hz_matched += 1

    if hz_matched > hz_match_min:

        if source_start == -1:
            source_start = x
            sample_matches_counts = []

        sample_matches_counts.append(hz_matched)

        if sample_matching > sample_length:

            start_time = (source_start * source_timing)

            match_counts_min = min(sample_matches_counts)
            match_counts_max = max(sample_matches_counts)

            if match_min == None or match_min > match_counts_min:
                match_min = match_counts_min

            if match_max == None or match_max < match_counts_max:
                match_max = match_counts_max

            print('')
            print('    FOUND @ {}'.format(start_time))
            print('      Min: {}'.format(match_counts_min))
            print('      Max: {}'.format(match_counts_max))
            # print(sample_matches_counts)
            print('')

            subprocess.call(['ffmpeg', '-loglevel', 'quiet', '-ss', str(start_time), '-i', source_path, '-t', str(sample_length * source_timing), ('05.stft-matching/%03d-%f.mp3' % (match_count, start_time))])

            sample_matching = sample_start
            source_start = -1
            match_count += 1

        else:

            sample_matching += 1

    elif sample_matching > sample_start:

        print('  Reset {} {}/{} @ {}'.format(sample_matching, hz_matched, hz_match_min, (source_start * source_timing)))

        sample_matching = sample_start
        x = source_start
        source_start = -1

        reset_count += 1

    x += 1

print('')
print('  Reset count: {}'.format(reset_count))
print('  Match count: {}'.format(match_count))
print('  Match min: {}'.format(match_min))
print('  Match max: {}'.format(match_max))
print('')

#--------------------------------------------------

print('Done')
print('')
sys.exit()
