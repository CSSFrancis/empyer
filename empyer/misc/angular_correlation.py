import numpy as np
from empyer.misc.image import bin_2d


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
    if mask is not None:
        image[mask] = 0
        if cut_off is not 0:
            mask = mask[cut_off:len(mask), :]

    if cut_off is not 0:
        image = image[cut_off:len(image), :]

    if binning is not 1:
        image = bin_2d(image, binning)
        if mask is not None:
            mask = bin_2d(mask, binning) != 0



    # fast method uses a FFT and is a process which is O(n) = n log(n)
    I_fft = np.fft.fft(image, axis=1)
    a = np.fft.ifft(I_fft * np.conjugate(I_fft), axis=1).real
    # this is to determine how many of the variables were non zero... This is really dumb.  but...
    # it works and I should stop trying to fix it (wreak it)
    if mask is not None:
        mask_boolean = ~mask  # inverting the boolean mask
        mask_fft = np.fft.fft(mask_boolean, axis=1)
        number_unmasked = np.fft.ifft(mask_fft*np.conjugate(mask_fft), axis=1).real
        number_unmasked[number_unmasked == 0] = 1  # get rid of divide by zero error for completely masked rows
        a = np.multiply(np.divide(a, number_unmasked), 720)

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
    """
    :param correlation: Taking the FFT of the angular correlation to find the symmetry present
    :param method: Right now this doesn't actually do anything but I want to add in other methods.
    :return: pow_spectrum: The resulting power spectrum from the angular correlation.  Gives indexes 0-180.
    """

    if method is "FFT":
        print(np.shape(correlation))
        pow_spectrum = np.fft.fft(correlation, axis=1).real
        pow_spectrum = np.power(pow_spectrum, 2)
    return pow_spectrum


def get_S_Q(r_theta_img,plot=False):

    # TODO: add in what the physical r is for the function...

    S_Q = [np.nansum(r) for r in r_theta_img]
    if plot:
        plt.plot(S_Q)
        plt.show()
    return S_Q

