
# Audio Detect

Identify when a sound effect is played multiple times in an audio file (e.g. an MP3).

Otherwise known as Audio Event Detection.

---

While the main script only really needs `scipy` and `numpy`; using the `requirements.txt` file will give you those dependencies and support the script that does visualisations:

    pip3 install -r requirements.txt

Then simply run the `process.py` script:

    python3 ./process.py ./example/config.py ./example/source.mp3;

The `config.py` file allows you to configure this - e.g. where the sample file(s) are specified.

The output will end with something like:

    Matches
     1.mp3 = 0:00:01.578957
     2.mp3 = 0:00:04.086712
     1.mp3 = 0:00:05.363810
     1.mp3 = 0:00:07.825125

If you want this data to be put in files, look at the 'output_title' config variable.

---

## Creating Samples

To create sample files, as in the small sound you are trying to find, run:

    python3 ./process.sample.py ./example/samples/ ./example/source.mp3 1 0.7 "1.59";

    python3 ./process.sample.py ./example/samples/ ./example/source.mp3 2 0.7 "4.08";

Arguments:

1) Path to samples folder
2) Path to source file
3) Sample number
4) Sample length, in seconds
5) Sample start time

This command will create files in 'img' and 'info' sub-folders - these files are are not needed, but they help keep a record of what the samples are, and how they look.

With the images, focus on the lower/orange one (this shows the data this script is working with).

The first blue line shows where the matching starts, and should be on the first set of dark squares. This is set a few frames after the first bit of noise, as each frame samples noise before/after it, which means that the first few are always wrong.

If you would rather create these files yourself, you can do something like:

    ffmpeg -ss 1.59 -t 0.7 -i ./example/source.mp3 -acodec copy -y ./example/samples/1b.mp3

---

## Background

This script uses a lot of ideas from "librosa" project, with some differences.

Rather than creating a "floating point time series" in this script via `librosa.load`, we rely on ffmpeg to get the PCM data - this is considerably faster for audio files that are over an hour long.

Rather than processing the whole file, then comparing via STFT (uses a lot of memory), do the same steps but in 31 block segments.

---

While identifying sound effects can be useful, so can the process of identifying who is talking:

https://github.com/ppwwyyxx/speaker-recognition
