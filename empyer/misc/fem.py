import numpy as np


def fem(r_theta_imgs, version="omega", binning=1, cut=40):
    """Calculated the variance among some image

    Parameters
    ----------
    r_theta_imgs: array_like
        polar images
    version: str
        The name of the FEM equation to use
    binning : int
        binning factor
    cut : int
        The cut off in pixels to not consider
    """
    imgs = r_theta_imgs
    imgs = imgs[:, cut // binning:len(imgs), :]
    if version is "omega":
        I_2_avg = np.mean(np.power(imgs, 2), axis=2)
        print(np.shape(I_2_avg))
        I_avg_2 = np.power(np.mean(imgs, axis=2), 2)
        print(np.shape(I_avg_2))
        int_vs_k = np.mean(np.divide(np.subtract(I_2_avg, I_avg_2), I_avg_2), axis=0)
        print(np.shape(int_vs_k))
    else:
        print(np.shape(r_theta_imgs))
        avg2 = np.mean(np.power(np.mean(r_theta_imgs, axis=1), 2), axis=0)
        print(np.shape(avg2))
        avg1 = np.power(np.mean(np. mean(r_theta_imgs, axis=1), axis=0), 2)
        print(np.shape(avg1))
        int_vs_k = np.divide(np.subtract(avg2, avg1), avg1)
        plt.plot(avg2)
        plt.plot(avg1)
        plt.show()
        plt.plot(int_vs_k)
        plt.show()
    return int_vs_k