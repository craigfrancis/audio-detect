
# Audio Detect

If you have a certain sound effect that's played multiple times in an audio file (e.g. an MP3), I'd like to identify when that happens.

Otherwise known as Audio Event Detection.

Create 1 or more sample files:

	ffmpeg -ss 1.59 -t 0.7 -i ./source.mp3 -acodec copy -y ./samples/1.mp3

Then run the `process.py` script:

	python ./process.py ./source.mp3 ./samples/ 0.75

---

This script uses a lot of ideas from "librosa" project, with some differences.

Rather than creating a "floating point time series" in this script via `librosa.load`, we rely on ffmpeg to get the PCM data - this is considerably faster for audio files over an hour long.

Rather than processing the whole file, then comparing via STFT (uses a lot of memory), do the same steps but in 31 block segments.

---

While identify sound effects can be useful, so can the process of identifying who is talking:

https://github.com/ppwwyyxx/speaker-recognition
