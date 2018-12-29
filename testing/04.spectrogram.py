#--------------------------------------------------
# After asking on StackoverFlow it was suggested
# that the audio should be analysed by frames,
# ideally with them overlapping.
#
# To understand this better, I'm first going to
# look at it via a spectrogram.
#--------------------------------------------------

import sys
import numpy as np
import librosa

import librosa.display
import matplotlib.pyplot as plt

#--------------------------------------------------

if len(sys.argv) == 2:
    sample_id = sys.argv[1]
else:
    print('Missing Sample ID')
    sys.exit()

sample_path = './03.series-big/sample-' + sample_id + '.mp3'

#--------------------------------------------------

print('Load file')

sample_series, sample_rate = librosa.load(sample_path)

#--------------------------------------------------

print('Parse data')

sample_data = librosa.stft(sample_series, hop_length=64)
sample_data = np.abs(sample_data)
sample_data = librosa.amplitude_to_db(sample_data, ref=np.max)

#--------------------------------------------------

print('Spectrogram')

plt.figure(figsize=(14, 5))
librosa.display.specshow(sample_data, sr=sample_rate, x_axis='time', y_axis='log', cmap='Reds')
plt.savefig('./04.spectrogram/sample-' + sample_id + '.png')
