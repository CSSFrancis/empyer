**EMpyer**

Empyer is an extension of the hyperspy_ package.  It provides additional functionality related to analyzing 4 and 5
dimensional data sets.  Especially STEM diffraction patterns from metallic glasses. Now Empyer can be loaded just
through loading hyperspy_.  That means that once you install EMpyer it will automatically register the new methods
and signals with the hyperspy_ package.

Documentation can be found hosted here_.

*Quick Start Guide:*


.. code:: bash

    $pip install empyer

.. code:: python

    import hyperspy.api as hs
    import matplotlib.pyplot as plt

    dif_signal = hs.load(file, signal_type ='diffraction_signal')
    dif_signal.plot()
    plt.show()

.. _hyperspy: https://github.com/hyperspy
.. _here: https://empyer.readthedocs.io/en/latest/