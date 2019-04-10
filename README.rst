**EMpyer**

Empyer is an extension of the hyperspy_ package.  It provides additional functionality related to analyzing 4 and 5
dimensional data sets.  Especially STEM diffraction patterns from metallic glasses.

*Quick Start Guide*


.. code:: bash

    $pip install empyer

.. code:: python

    import empyer
    import matplotlib.pyplot as plt
    dif_signal = empyer.load(file, signal_type ='diffraction_signal')
    dif_signal.plot()
    plt.show()
    print 8/2

.. _hyperspy: https://github.com/hyperspy