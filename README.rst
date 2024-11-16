omf
***

.. image:: https://img.shields.io/pypi/v/mira-omf.svg
    :target: https://pypi.python.org/pypi/mira-omf
    :alt: Latest PyPI version

.. image:: https://readthedocs.org/projects/omf/badge/?version=stable
    :target: http://omf.readthedocs.io/en/stable/
    :alt: Documentation

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/MiraGeoscience/omf/blob/main/LICENSE
    :alt: MIT license

.. image:: https://github.com/MiraGeoscience/omf/actions/workflows/pytest-windows.yml/badge.svg
    :target: https://github.com/MiraGeoscience/omf/actions/workflows/pytest-windows.yml
    :alt: pytest


Version: 3.3.0-beta.1

API library for Open Mining Format, a new standard for mining data backed by
the `Global Mining Standards & Guidelines Group <https://gmggroup.org/>`_.


.. warning::

    This is a fork created by Mira Geoscience for interoperability with the
    geoh5 file format.

Why?
----

An open-source serialization format and API library to support data interchange
across the entire mining community.

Scope
-----

This library provides an abstracted object-based interface to the underlying
OMF serialization format, which enables rapid development of the interface while
allowing for future changes under the hood.

Goals
-----

- The goal of Open Mining Format is to standardize data formats across the
  mining community and promote collaboration
- The goal of the API library is to provide a well-documented, object-based
  interface for serializing OMF files

Alternatives
------------

OMF is intended to supplement the many alternative closed-source file formats
used in the mining community.

Connections
-----------

This library makes use of the `properties <https://github.com/seequent/properties>`_
open-source project, which is designed and publicly supported by
`Seequent <https://seequent.com>`_.

Connection to the geoh5 format makes use of `geoh5py <https://geoh5py.readthedocs.io/>`_
publicly supported by `Mira Geoscience <https://mirageoscience.com/>`_

Installation
------------

To install the repository, ensure that you have
`pip installed <https://pip.pypa.io/en/stable/installing/>`_ and run:

.. code:: bash

    pip install omf

Or from `github <https://github.com/GMSGDataExchange/omf>`_:

.. code:: bash

    git clone https://github.com/GMSGDataExchange/omf.git
    cd omf
    pip install -e .


License
^^^^^^^
MIT License

Copyright (c) 2024 Mira Geoscience

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Third Party Software
^^^^^^^^^^^^^^^^^^^^
The mira-omf Software may provide links to third party libraries or code (collectively "Third Party Software")
to implement various functions. Third Party Software does not comprise part of the Software.
The use of Third Party Software is governed by the terms of such software license(s).
Third Party Software notices and/or additional terms and conditions are located in the
`THIRD_PARTY_SOFTWARE.rst`_ file.

.. _THIRD_PARTY_SOFTWARE.rst: THIRD_PARTY_SOFTWARE.rst
