#--------------------------------------------------
# Trying to find a pattern via librosa.load only.
#
# The sample file is created via ffmpeg 'copy' by:
#
#   ffmpeg -ss 1.59 -i ./source-256.mp3 -t 0.7 -acodec copy -y ./03.series-big/sample-1a.mp3
#   ffmpeg -ss 5.38 -i ./source-256.mp3 -t 0.7 -acodec copy -y ./03.series-big/sample-2a.mp3
#   ffmpeg -ss 7.85 -i ./source-256.mp3 -t 0.7 -acodec copy -y ./03.series-big/sample-3a.mp3
#
# While all 3 samples sound the same, there are
# very slight variations, so each one will only
# be found once:
#
#   `python 03.series-big.py 1a` = FOUND @ 1.62013605442
#   `python 03.series-big.py 2a` = FOUND @ 5.40489795918
#   `python 03.series-big.py 3a` = FOUND @ 7.88816326531
#
# If it's not an exact copy, there it's not found:
#
#   ffmpeg -ss 1.59 -i ./source-256.mp3 -t 0.7 -y ./03.series-big/sample-1b.mp3
#   ffmpeg -ss 5.38 -i ./source-256.mp3 -t 0.7 -y ./03.series-big/sample-2b.mp3
#   ffmpeg -ss 7.85 -i ./source-256.mp3 -t 0.7 -y ./03.series-big/sample-3b.mp3
#
#   `python 03.series-big.py 1b` = Not found
#   `python 03.series-big.py 2b` = Not found
#   `python 03.series-big.py 3b` = Not found
#
#--------------------------------------------------

import sys
import numpy
import librosa

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

print('Load files')

source_series, source_rate = librosa.load(source_path)
sample_series, sample_rate = librosa.load(sample_path)

#--------------------------------------------------

# print('Round series')
#
# source_series = numpy.around(source_series, decimals=5)
# sample_series = numpy.around(sample_series, decimals=5)

#--------------------------------------------------

# print('Save series')
#
# with open('03.series-big/source.txt', 'w') as f:
#   for value in source_series:
#     f.write("%s\n" % value)
#
# with open('03.series-big/sample-' + sample_id + '.txt', 'w') as f:
#   for value in sample_series:
#     f.write("%s\n" % value)

#--------------------------------------------------

print('Save waveshow')

plt.figure()
plt.subplot(3, 1, 1)
librosa.display.waveshow(source_series, sr=source_rate)
plt.savefig('03.series-big/source.png')

plt.figure()
plt.subplot(3, 1, 1)
librosa.display.waveshow(sample_series, sr=sample_rate)
plt.savefig('03.series-big/sample-' + sample_id + '.png')

#--------------------------------------------------

print('Sample start')

sample_start = 0
for sample_id, sample_value in enumerate(sample_series):
    if sample_value != 0:
        sample_start = sample_id
        break

sample_start += 1000 # The leading ~300 elements are different for some reason.

print('')
print('  {} of {}'.format(sample_start, len(sample_series)))
print('')

#--------------------------------------------------

print('Process series')

source_start = -1
source_length = len(source_series)
sample_matching = sample_start
sample_length = (len(sample_series) - sample_start)

x = 0
while x < source_length:

    if source_series[x] == sample_series[sample_matching]:

        if source_start == -1:
            source_start = x

        if sample_matching > sample_length:
            print('')
            print('    FOUND @ {}'.format(float(source_start) / source_rate))
            print('')
            sample_matching = sample_start
            source_start = -1
        else:
            sample_matching += 1

    elif sample_matching > sample_start:

        print('  Reset {} @ {}'.format(sample_matching, float(source_start) / source_rate))

        sample_matching = sample_start
        x = source_start
        source_start = -1

    x += 1

#--------------------------------------------------

print('Done')
print('')
sys.exit()
