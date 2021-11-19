.. RMP Framework documentation master file, created by
   sphinx-quickstart on Fri Oct  1 15:04:46 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RMP Framework's documentation!
=========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

RMP Framework
=============

**RMP Framework** is a GUI framework for robot motion planning written in Python.
Includes projects for simulation and verification of RMP under various scenarios.

Prerequisites
=============

All projects in this repository run on Python 3.9, and require the ``PyQt5``
module. The module can be installed by running the command:

``pip install PyQt5``

Some projects further require the ``networkx`` module which can similarly be
installed by running:

``pip install networkx``

And most projects require a specific compiled library of CGAL python bindings
(CGALPY).
For Windows 64 bit, some precompiled libraries are already included under the
``cgal_bindings`` folder.

In order to run ``build.bat`` scripts (see below), the ``pyinstaller``
module is also required:

``pip install pyinstaller``

Packing and Building
====================

To ease distribution on Windows machines, each project includes two .bat
scripts, ``pack.bat`` and ``build.bat``.
Running ``pack.bat`` generates a ``package`` folder into which all the
required folders for running the project are copied. Running ``build.bat``
generates a standalone distribution for Windows machines with no further
dependencies inside a ``dist`` folder.  The project's code will also be
hidden in this case, and the project would be run via an executable.

Projects overview
=================

Below is a short overview of the projects:

* **Interface**
   Provides a minimal interface for passing inputs to an external script and
   drawing 2D graphics from that script.


* **Multi Robot Motion Planning - Disc**
   Provides simulation and verification for simultaneous motion of multiple
   disc robots in scenes consisting of polygonal obstacles.

* **Multi Robot Motion Planning - Polygon**
   Provides simulation and verification for simultaneous motion of multiple
   polygonal robots in scenes consisting of polygonal obstacles.

* **Robot Motion Planning Workshop**
   Provides an environment for running a turn based robot motion planning game in
   which two resource limited teams need to plan motions for multiple disc robots
   among polygonal scenes in order to reach some goals.

* **Rod**
   Provides simulation and verification for the motion of a rod robot involving
   translation and rotation in scenes consisting of polygonal obstacles.

* **Scene Designer**
   Provides a graphical tool to assist with designing polygonal scenes.