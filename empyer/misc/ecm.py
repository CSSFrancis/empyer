import numpy as np


def ecm(data):
    """Calculates the correlation between some pixel and the same pixel at time t+tau
    Make sure that the data that is fed to the ecm method is the time data at some pixel value.

    Parameters
    ----------
    data: array_like
        pixel values for some x,y position over a time series
    """
    # fast method uses a FFT and is a process which is O(n) = n log(n)
    data_length = len(data)
    data_fft = np.fft.fft(data)
    time_correlation = np.fft.ifft(data_fft * np.conjugate(data_fft)).real
    #print(len(time_correlation))
    # normalizing the data
    norm_correlation = time_correlation/data_length
    norm_correlation = norm_correlation/np.mean(data)**2
    return norm_correlation
