#--------------------------------------------------
# Basic matching... kind of works, but very slow
#--------------------------------------------------

import sys
import numpy as np
import librosa

import os
import glob
import subprocess

import librosa.display
import matplotlib.pyplot as plt

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
hz_match_min = int(sample_height * 0.70) # x% of 1025

print('')
print('  Diff Match: {}'.format(hz_diff_match))
print('  Match Min: {}'.format(hz_match_min))
print('')

#--------------------------------------------------

print('Process series')

source_start = -1
sample_matching = sample_start
sample_matches_counts = []

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
            print('')
            print('    FOUND @ {}'.format(source_start * source_timing))
            print(sample_matches_counts)
            print('')
            sample_matching = sample_start
            source_start = -1
        else:
            sample_matching += 1

    elif sample_matching > sample_start:

        print('  Reset {} {}/{} @ {}'.format(sample_matching, hz_matched, hz_match_min, (source_start * source_timing)))

        sample_matching = sample_start
        x = source_start
        source_start = -1

    x += 1

#--------------------------------------------------

print('Done')
print('')
sys.exit()
