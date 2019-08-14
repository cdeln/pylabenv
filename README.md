# PyLabEnv
A small package that defines commonly used numerical functions and data structures for scientifc computing.
Created to simulate something that feels like a Matlab-like environment in Python.
Goes well together with the [Spyder IDE](https://www.spyder-ide.org/).

## Installation
* `make`
* `make install`

## Usage
### In a script
`from pylabenv import *`
### Interactively
In a terminal, run `pylab` or `python -i -c "from pylabenv import *"`

## Common problems
If you get problems with Qt, then try to uninstall *libqt5x11extras5* ss shown in [this thread](https://github.com/skvark/opencv-python/issues/46)
