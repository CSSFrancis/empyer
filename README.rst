**EMpyer**

Empyer is an extension of the hyperspy_ package.  It provides additional functionality related to analyzing 4 and 5
dimensional data sets.  Especially STEM diffraction patterns from metallic glasses.

Documenation can be found hosted here_.

*Quick Start Guide:*


.. code:: bash

    $pip install empyer

.. code:: python

    import empyer
    import matplotlib.pyplot as plt

    dif_signal = empyer.load(file, signal_type ='diffraction_signal')
    dif_signal.plot()
    plt.show()

.. _hyperspy: https://github.com/hyperspy
.. _here: https://empyer.readthedocs.io/en/latest/