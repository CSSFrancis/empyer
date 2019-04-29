Analyzing Angular Correlations
====================================

While angular correlations can be easily (and quickly calculated) they are highly susceptible to even small
misalignment in the microscope.  Additionally, thick samples which are more dominated by dynamical scattering, render
the data collected largely useless.

Starting with the equation for the angular correlation is given by:


.. math::

   C(\phi,k,n)= \frac{ <I(\theta,k,n)*I(\theta+\phi,k,n)>_\theta-<I(\theta,k,n)>^2}{<I(\theta,k,n)>^2}

Where :math:`\theta` is the entire :math:`2\pi` radians the that :math:`\phi` (the angle of correlation) is averaged
over. k is the radius of the reciporical space vector and n is the diffraction pattern number (assuming the correlation
is being calculated for a series of diffraction patterns)

While the Electron Microscope community has decided to use the terminology of angular correlations, what is being
calculated is in actuality the self-correlation as a function of angle instead of time.

As result efficient methods for convolution can be applied in which the actual computation occurring is:

.. math::

   C(\phi,k,n)=\frac{IFFT[FFT(I(\theta,k,n))_\theta * Conj(FFT(I(\theta,k,n)))_\theta]}{<I(\theta,k,n)>^2}

Which speeds up the calculation more than 100x