Installation
============

Requirements
------------

* Python ≥ 3.10
* A working OpenCV install (the wheel ``opencv-python`` is fine)
* Qt offscreen libraries for headless test runs (Linux only):
  ``libegl1 libgl1 libxkbcommon-x11-0 libdbus-1-3 libxcb-cursor0``

From source
-----------

.. code-block:: bash

   git clone https://github.com/zerotonin/pyLACE.git
   cd pyLACE
   pip install -e ".[dev]"

The ``[dev]`` extra adds ``pytest``, ``pytest-cov`` and ``ruff``.

Verifying the install
---------------------

.. code-block:: bash

   pytest
   pylace-detect --help

Running the headless test suite (no display)
--------------------------------------------

.. code-block:: bash

   QT_QPA_PLATFORM=offscreen pytest

This is what the ``tests`` GitHub Actions workflow does on every
push.
