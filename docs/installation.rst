.. highlight:: shell

============
Installation
============

This package has not been published.

From sources
------------

The sources for gridverse-experiments can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/samkatt/gridverse_experiments

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/samkatt/gridverse_experiments/tarball/main

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install

Or:

.. code-block:: console

    $ python -m pip install .

Dependencies
------------

This package requires dependencies not easily available through pip:

- `Online POMDP planning <https://github.com/samkatt/online-pomdp-planning>`_ 
- `POMDP belief tracking <https://github.com/samkatt/pomdp-belief-tracking>`_ 
- `Gym Gridverse <https://github.com/abaisero/gym-gridverse>`

Please install them per their instructions.

.. _Github repo: https://github.com/samkatt/gridverse_experiments
.. _tarball: https://github.com/samkatt/gridverse_experiments/tarball/main
