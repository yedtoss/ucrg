# Description
This repository contain the code to replicate experiments done for the paper: "A Novel Individually Rational Objective In Multi-Agent
Multi-Armed Bandits: Algorithms and Regret Bounds". Implementation is done using Python.

## Requirements
It requires and has been tested on Python 3.


## Installation

Run the following to install python dependencies
```bash
pip install -r requirements.txt
```

There is an additional dependency (gambit). For that we need Gambit with python 3 support.
So gambit released after (11 Jan 2019, See https://github.com/gambitproject/gambit/issues/203).

To install gambit with python binding You need to get the latest version here: https://github.com/gambitproject/gambit
And follow installation direction from git repository here https://github.com/gambitproject/gambit/blob/master/INSTALL

Here are instructions to install on Ubuntu (along with fixes to installation procedure)

```bash
git clone git://github.com/gambitproject/gambit.git
cd gambit
mkdir -p m4 # This fix a issue where aclocal fail to run if m4 directory doesn't exist.
aclocal
libtoolize
automake --add-missing
autoconf
./configure
make
sudo make install

#(http://www.gambit-project.org/gambit16/16.0.0/build.html#build-python)
#Building the Python api (which supportes Python 3 since 11 Jan 2019)
cd src/python
python setup.py build
sudo python setup.py install

For running python test (Make sure nose package is installed)
cd src/python/gambit/tests
nosetests
```


## Running
To run the experiments execute
```bash
python test.py
```
The figures of the experiments will appear in figures/*.png

## License
MIT

