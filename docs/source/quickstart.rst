Quickstart
==========

Welcome to the Quick Start Page!

Empyer is an extension of the hyperspy_ package.  It provides additional functionality related to analyzing 4 and 5
dimensional data sets.  Especially STEM diffraction patterns from metallic glasses.


Downloading EMpyer is easy.  You can download the latest version of EMpyer from PyPi using pip.

.. code:: bash

    $pip install empyer

Assuming you are looking at Diffraction patterns from a STEM you can easily view the data by just sending the plot
command to a _diffraction_signal object.  Utalizing the plotting and loading abilities from hyperspy_ the signal will be
shown.

.. code:: python

    import empyer
    import matplotlib.pyplot as plt

    dif_signal = empyer.load(file, signal_type ='diffraction_signal')
    dif_signal.plot(norm='log')
    plt.show()

.. _hyperspy: https://github.com/hyperspy

