#--------------------------------------------------
# Find a pattern of numbers, in a bigger set.
#--------------------------------------------------

import sys
import numpy
import librosa

#--------------------------------------------------

print('')
print('Create source')
print('')

source_series = [0, 1, 2, 9, 6, 0, 0, 9, 9, 1, 3, 4, 5, 6, 9, 1, 3, 2, 1, 7, 4, 3, 2, 9, 1, 3, 4, 1, 1]
sample_series = [0, 0, 9, 1, 3, 4]

#--------------------------------------------------

print('Sample start')
print('')

sample_start = 0
for sample_id, sample_value in enumerate(sample_series):
    if sample_value != 0:
        sample_start = sample_id
        break

print('  {}'.format(sample_start))
print('')

#--------------------------------------------------

print('Process series')
print('')

source_start = -1
source_length = len(source_series)
sample_matching = sample_start
sample_length = (len(sample_series) - sample_start)

x = 0
while x < source_length:

    match = (source_series[x] == sample_series[sample_matching])

    print('  {} / {} ... {} == {} ({})'.format(x, sample_matching, source_series[x], sample_series[sample_matching], ('Y' if match else '-')))

    if match:

        if source_start == -1:
            source_start = x

        if sample_matching > sample_length:
            print('')
            print('    FOUND @ {}'.format(source_start))
            print('')
            sample_matching = sample_start
            source_start = -1
        else:
            sample_matching += 1

    elif sample_matching > sample_start:

        print('  Reset {}'.format(sample_matching))

        sample_matching = sample_start
        x = source_start
        source_start = -1

    x += 1

#--------------------------------------------------

print('')
print('Done')
print('')
sys.exit()
