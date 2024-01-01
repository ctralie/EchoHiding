import numpy as np
import matplotlib.pyplot as plt

hann = lambda win: 0.5*(1-np.cos(2*np.pi*np.arange(win)/win))

def get_hann_mixer(L, b):
    h = hann(L*2)
    m = np.zeros(L*b.size)
    for i, bi in enumerate(b):
        if bi == 1:
            if i == 0:
                m[0:L//2] = 1
            else:
                m[i*L-L//2:i*L+L//2] += h[0:L]
            if i == b.size-1:
                m[i*L+L//2:(i+1)*L] = 1
            else:
                m[i*L+L//2:(i+1)*L+L//2] += h[L::]
    return m

def echo_hide(x, L, b, delta0=50, delta1=75, alpha=0.4):
    """
    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    L: int
        Segment length
    b: ndarray(K)
        Binary samples
    delta0: int
        Delay of a zero echo
    delta1: int
        Delay of a one echo
    alpha: float
        Amplitude of echo
    
    Returns
    -------
    ndarray(b*hop)
        Audio samples with echo hidden
    """
    if x.size < L*b.size:
        raise ValueError("Error: Audio size {} is not long enough to hold {} bits of length {}".format(x.size, b.size, L))
    x0  = np.array(x[0:-delta1])
    x0 += alpha*x[delta0:delta0+x0.size]
    x1  = x[0:-delta1] + alpha*x[delta1::]
    m = get_hann_mixer(L, b)
    return m*x1[0:m.size] + (1-m)*x0[0:m.size]

def extract_echo_bits(y, L, delta0=50, delta1=75):
    """
    Parameters
    ----------
    y: ndarray(N)
        Audio samples
    L: int
        Segment length
    delta0: int
        Delay of a zero echo
    delta1: int
        Delay of a one echo
    
    Returns
    -------
    ndarray(N//L)
        Array of estimated bits
    """
    h = hann(L*2)
    yp = np.pad(y, (L//2, L//2))
    T = (yp.size-L*2)//L+1
    n_even = yp.size//(L*2)
    n_odd = T-n_even
    Y = np.zeros((T, L*2))
    Y[0::2, :] = np.reshape(yp[0:n_even*L*2], (n_even, L*2))
    Y[1::2, :] = np.reshape(yp[L:L+n_odd*L*2], (n_odd, L*2))
    Y = Y*h[None, :]
    F = np.abs(np.fft.rfft(Y, axis=1))
    F = np.fft.irfft(np.log(F+1e-8), axis=1)
    return F[:, delta1] > F[:, delta0]

def get_odg_distortion(x, y, sr, cleanup=True):
    """
    A wrapper around GstPEAQ for computing objective measurements
    of pereceived audio quality

    Parameters
    ----------
    x: ndarray(N)
        Reference audio
    y: ndarray(N)
        Test audio
    sr: int
        Sample rate
    
    Returns
    -------
    odg: float
        Objective difference grade
             0 - impairment imperceptible
            -1 - impairment perceptible but not annoying
            -2 - impairment slightly annoying
            -3 - impairment annoying
            -4 - impairment very annoying
    di: float
        Distortion index
    """
    from scipy.io import wavfile
    import os
    from subprocess import check_output
    ref = np.array(x*32768, dtype=np.int16)
    wavfile.write("ref.wav", sr, ref)

    test = np.array(y*32768, dtype=np.int16)
    wavfile.write("test.wav", sr, test)
    res = check_output(["peaq", "--advanced", "ref.wav", "test.wav"])
    odg = float(str(res).split("\\n")[0].split()[-1])
    di = float(str(res).split("\\n")[1].split()[-1])
    if cleanup:
        os.remove("ref.wav")
        os.remove("test.wav")
    return odg, di

def get_mp3_encoded(x, sr, bitrate):
    """
    Parameters
    ----------
    x: ndarray(N, dtype=float)
        Mono audio samples in [-1, 1]
    sr: int
        Sample rate
    bitrate: int
        Number of kbits per second to use in the mp3 encoding
    """
    import subprocess
    import os
    from scipy.io import wavfile
    x = np.array(x*32768, dtype=np.int16)
    wavfile.write("temp.wav", sr, x)
    if os.path.exists("temp.mp3"):
        os.remove("temp.mp3")
    subprocess.call(["ffmpeg", "-i", "temp.wav","-b:a", "{}k".format(bitrate), "temp.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    os.remove("temp.wav")
    subprocess.call(["ffmpeg", "-i", "temp.mp3", "temp.wav"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    os.remove("temp.mp3")
    _, y = wavfile.read("temp.wav")
    os.remove("temp.wav")
    return y/32768