#ae

Python acoustic emission tools.

##Features
* read *.wfs files
* read *.sdcf files
* event extractor
* interactive decimated waveform plotting
* optimized for files bigger than RAM

##[Example](http://nbviewer.ipython.org/github/jove1/ae/blob/master/doc/example.ipynb)
```python
>>> import ae
>>> f = ae.open("M5.wfs")
>>> print f.size
642798592
>>> f.plot()
```
![Graph](doc/view.png)
```python
>>> events = f.get_events(0.02)
>>> print events.size
6161
>>> ae.hist(event.maxima)
```
![Graph](doc/hist.png)

##Install
###Dependencies: 
* [python](http://python.org)
* [numpy](http://scipy.org)
* [matplotlib](http://matplotlib.org)

###Linux:
```
git clone https://github.com/jove1/ae.git
python setup.py install
```

###Windows:
Get windows installers of [python](http://python.org/downloads/windows/), [scipy-stack](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy-stack) and [ae](https://pypi.python.org/pypi/ae/0.2) itself.

