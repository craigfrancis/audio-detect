import sys
import os
import glob
import subprocess
import numpy as np
import scipy
import scipy.signal
import datetime


matches = [[2, 279.96299319727893, False], [2, 1206.2998639455782, False], [3, 2151.7931972789115, False], [2, 3154.6862585034014, False], [0, 3842.4845351473923, True], [2, 3855.185850340136, False], [2, 4776.483990929705, False], [2, 5706.071655328798, False], [2, 6758.191020408163, False], [0, 7415.315736961451, True], [2, 7428.017052154195, False], [2, 8290.382947845805, False], [2, 9374.17433106576, False], [2, 10164.21006802721, False]]

# Matches
#  2 = 0:04:39.962993 @ 279.962993197
#  2 = 0:20:06.299864 @ 1206.29986395
#  3 = 0:35:51.793197 @ 2151.79319728
#  2 = 0:52:34.686259 @ 3154.6862585
#  0 = 1:04:02.484535 @ 3842.48453515 - Dupe
#  2 = 1:04:15.185850 @ 3855.18585034
#  2 = 1:19:36.483991 @ 4776.48399093
#  2 = 1:35:06.071655 @ 5706.07165533
#  2 = 1:52:38.191020 @ 6758.19102041
#  0 = 2:03:35.315737 @ 7415.31573696 - Dupe
#  2 = 2:03:48.017052 @ 7428.01705215
#  2 = 2:18:10.382948 @ 8290.38294785
#  2 = 2:36:14.174331 @ 9374.17433107
#  2 = 2:49:24.210068 @ 10164.210068

meta_path = 'test.txt';
meta_title = 'Test';

samples = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]];


f = open(meta_path, 'w')
f.write(';FFMETADATA1\n')
f.write('title=' + meta_title + '\n')
f.write('\n')
k = 0
last_time = 0
last_sample = 'N/A'
for match in matches:
    if match[2] == False: # Not a dupe
        k += 1
        end_time = int(round(match[1]))
        f.write('[CHAPTER]\n')
        f.write('TIMEBASE=1/1000\n')
        f.write('START=' + str(last_time * 1000) + '\n')
        f.write('END=' + str(end_time * 1000) + '\n')
        f.write('title=Chapter ' + str(k) + '\n')
        f.write('#human-start=' + str(str(datetime.timedelta(seconds=last_time))) + '\n')
        f.write('#human-end=' + str(str(datetime.timedelta(seconds=end_time))) + '\n')
        f.write('#sample=' + str(last_sample) + '\n')
        f.write('\n')
        last_time = end_time
        last_sample = samples[match[0]][2]
if last_time > 0:
    k += 1
    end_time = int(round((float(source_frame_end) * hop_length) / sample_rate))
    f.write('[CHAPTER]\n')
    f.write('TIMEBASE=1/1000\n')
    f.write('START=' + str(last_time * 1000) + '\n')
    f.write('END=' + str(end_time * 1000) + '\n')
    f.write('title=Chapter ' + str(k) + '\n')
    f.write('#human-start=' + str(str(datetime.timedelta(seconds=last_time))) + '\n')
    f.write('#human-end=' + str(str(datetime.timedelta(seconds=end_time))) + '\n')
    f.write('#sample=' + str(last_sample) + '\n')
    f.write('\n')
f.close()