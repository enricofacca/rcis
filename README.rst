========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/rcis/badge/?style=flat
    :target: https://rcis.readthedocs.io/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/enricofacca/rcis/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/enricofacca/rcis/actions

.. |requires| image:: https://requires.io/github/enricofacca/rcis/requirements.svg?branch=main
    :alt: Requirements Status
    :target: https://requires.io/github/enricofacca/rcis/requirements/?branch=main

.. |codecov| image:: https://codecov.io/gh/enricofacca/rcis/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://codecov.io/github/enricofacca/rcis

.. |version| image:: https://img.shields.io/pypi/v/rcis.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/rcis

.. |wheel| image:: https://img.shields.io/pypi/wheel/rcis.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/rcis

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/rcis.svg
    :alt: Supported versions
    :target: https://pypi.org/project/rcis

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/rcis.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/rcis

.. |commits-since| image:: https://img.shields.io/github/commits-since/enricofacca/rcis/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/enricofacca/rcis/compare/v0.0.0...main



.. end-badges

Framework for iterative solver via reverse communication

* Free software: GNU Lesser General Public License v3 or later (LGPLv3+)

Installation
============

::

    pip install rcis

You can also install the in-development version with::

    pip install https://github.com/enricofacca/rcis/archive/main.zip


Documentation
=============


https://rcis.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
