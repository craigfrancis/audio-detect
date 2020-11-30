#--------------------------------------------------

import sys
import os
import glob
import subprocess

#--------------------------------------------------

ffmpeg_path = 'ffmpeg' # On Windows, you will probably need this to be \path\to\ffmpeg.exe

if len(sys.argv) > 4:
    samples_folder = sys.argv[1]
    source_path = sys.argv[2]
    sample_number = sys.argv[3]
else:
    print('Missing arguments')
    sys.exit()

if len(sys.argv) >= 6:
    source_length = sys.argv[4]
    source_timestamp = sys.argv[5]
else:
    source_length = None
    source_timestamp = None

source_path_split = os.path.splitext(source_path)
sample_path = os.path.join(samples_folder, sample_number + source_path_split[1]);
sample_path_split = os.path.split(sample_path)
sample_ext_split = os.path.splitext(sample_path_split[1])

#--------------------------------------------------

if source_timestamp:
    print(source_timestamp)
    subprocess.call([ffmpeg_path, '-ss', source_timestamp, '-t', source_length, '-i', source_path, '-acodec', 'copy', '-y', sample_path])

#--------------------------------------------------

f = open(os.path.join(sample_path_split[0], 'info', sample_ext_split[0] + '.txt'), 'w')
f.write('source: ' + source_path + '\n')
f.write('timestamp: ' + source_timestamp + '\n')
f.write('length_seconds: ' + source_length + '\n')
f.close()

#--------------------------------------------------

samples_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'process.samples.py')

subprocess.call(['python3', samples_script, samples_folder])
