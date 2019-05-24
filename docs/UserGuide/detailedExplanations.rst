Advanced Guide
====================================

This section goes further into the methods for correcting and optimizing your data.  While ideally most of these
functions would work with minimal human interaction the reality of the situation is that whatever can go wrong will
inevitably go wrong.

So the purpose of this tutorial is to show some of the most common errors and problems with analysis and how to solve
these problems.

**Advanced Loading**

Starting at the very beginning, the first thing you need to figure out the proper way to load your signal.  While
hyperspy is very good at loading a variety of signals.  Sometimes there is a bit of difficultly in loading the data in
a way that makes sense.  I have wrote some scripts_ which load emi, tiff and .mrc files.  While they might not be
incredibly useful they are a good start for how to efficiently package microscope outputs into .hdf5 files.

Most of the work he is setting up your axes correctly, which while not the hardest thing to do, will save many headaches
down the road.

.. image:: individualAngularCorrelation.png

**Advanced Image Registration**


.. _scripts: https://github.com/hyperspy
