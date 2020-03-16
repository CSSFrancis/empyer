import numpy as np
from empyer.misc.image import bin_2d
import matplotlib.pyplot as plt


def angular_correlation(r_theta_img, mask=None, binning=1, cut_off=0, normalize=True):
    """A program that takes a 2d image and then preforms an angular correlation on the image.
    Parameters
    ----------
    r_theta_img: array_like
        The index of the axes
    mask: boolean array
        The name of the axis
    binning : int
        binning factor
    cut_off : int
        The cut off in pixels to
    normalize: bool
        Subtract <I(\theta)>^2 and divide by <I(\theta)>^2
    """
    image = r_theta_img
    m = mask
    print(np.shape(m))
    if m is not None:
        mask_boolean = ~ m  # inverting the boolean mask
        mask_fft = np.fft.fft(mask_boolean, axis=1)
        number_unmasked = np.fft.ifft(mask_fft*np.conjugate(mask_fft), axis=1).real
        number_unmasked[number_unmasked < 1] = 1  # get rid of divide by zero error for completely masked rows
        image[m] = 0
        if cut_off is not 0:
            number_unmasked = number_unmasked[cut_off:len(number_unmasked), :]

    if cut_off is not 0:
        image = image[cut_off:len(image), :]

    if binning is not 1:
        image = bin_2d(image, binning)
        if m is not None:
            m = bin_2d(m, binning) != 0

    # fast method uses a FFT and is a process which is O(n) = n log(n)
    I_fft = np.fft.fft(image, axis=1)
    a = np.fft.ifft(I_fft * np.conjugate(I_fft), axis=1).real
    # this is to determine how many of the variables were non zero... This is really dumb.  but...
    # it works and I should stop trying to fix it (wreak it)
    if m is not None:
        a = np.multiply(np.divide(a, np.transpose(number_unmasked)), 720)

    if normalize:
        a_prime = np.zeros(np.shape(a))
        for i, row in enumerate(a):
            row_mean = np.mean(row)
            if row_mean == 0:
                normalized_row = np.divide(np.subtract(row, row_mean), 1)
            else:
                normalized_row = np.divide(np.subtract(row, row_mean), row_mean)
            a_prime[i, :] = normalized_row
            a = a_prime
    return a


def power_spectrum(correlation, method="FFT"):
    """Take the power spectrum for some correlation.  Takes the FFT of the correlation

    Parameters
    --------------
    correlation: array-like
        Taking the FFT of the angular correlation to find the symmetry present
    method: str ("FFT")
        Right now this doesn't actually do anything but I want to add in other methods.

    Returns
    -----------
    pow_spectrum: array-like
        The resulting power spectrum from the angular correlation.  Gives indexes 0-180.
    """

    if method is "FFT":
        pow_spectrum = np.fft.fft(correlation, axis=1).real
        pow_spectrum = np.power(pow_spectrum, 2)
    return pow_spectrum
