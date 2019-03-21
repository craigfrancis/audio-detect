#--------------------------------------------------

# np.set_printoptions(threshold=np.nan)

dtype = np.complex64
n_fft=2048
hz_count = int(1 + n_fft // 2) # 1025 (Hz buckets)
win_length = n_fft
hop_length = int(win_length // 4)
# hop_length = 64
sample_rate = 22050
sample_crop_start = 3
sample_crop_end = 2

match_any_sample = True

#--------------------------------------------------

def pcm_data(path, sample_rate):

    devnull = open(os.devnull)
    proc = subprocess.Popen(['ffmpeg', '-i', path, '-f', 's16le', '-ac', '1', '-ar', str(sample_rate), '-'], stdout=subprocess.PIPE, stderr=devnull)
    devnull.close()

    scale = 1./float(1 << ((8 * 2) - 1))
    y = scale * np.frombuffer(proc.stdout.read(), '<i2').astype(np.float32)

    return y

#--------------------------------------------------

def stft_raw(series, sample_rate, win_length, hop_length, hz_count, dtype):

    #--------------------------------------------------
    # Config

    window = 'hann'
    pad_mode='reflect'

    #--------------------------------------------------
    # Get Window

    fft_window = scipy.signal.get_window(window, win_length, fftbins=True)

    #--------------------------------------------------
    # Pad the window out to n_fft size... Wrapper for
    # np.pad to automatically centre an array prior to padding.

    axis = -1

    n = fft_window.shape[axis]

    lpad = int((n_fft - n) // 2)

    lengths = [(0, 0)] * fft_window.ndim
    lengths[axis] = (lpad, int(n_fft - n - lpad))

    if lpad < 0:
        raise ParameterError(('Target size ({:d}) must be at least input size ({:d})').format(n_fft, n))

    fft_window = np.pad(fft_window, lengths, mode='constant')

    #--------------------------------------------------
    # Reshape so that the window can be broadcast

    fft_window = fft_window.reshape((-1, 1))

    #--------------------------------------------------
    # Pad the time series so that frames are centred

    series = np.pad(series, int(n_fft // 2), mode=pad_mode)

    #--------------------------------------------------
    # Window the time series.

        # Compute the number of frames that will fit. The end may get truncated.
    frame_count = 1 + int((len(series) - n_fft) / hop_length) # Where n_fft = frame_length

        # Vertical stride is one sample
        # Horizontal stride is `hop_length` samples
    frames_data = np.lib.stride_tricks.as_strided(series, shape=(n_fft, frame_count), strides=(series.itemsize, hop_length * series.itemsize))

    #--------------------------------------------------
    # how many columns can we fit within MAX_MEM_BLOCK

    MAX_MEM_BLOCK = 2**8 * 2**10
    n_columns = int(MAX_MEM_BLOCK / (hz_count * (dtype(0).itemsize)))

    #--------------------------------------------------
    # Return

    return (frames_data, fft_window, n_columns)
