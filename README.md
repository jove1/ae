ae
==

Python acoustic emission tools.

Features
--------

* read *.wfs files
* read *.sdcf files
* event extractor
* interactive decimated waveform plotting
* optimized for files bigger than RAM

[Example](http://nbviewer.ipython.org/github/jove1/ae/blob/master/doc/example.ipynb)
------------------------------------------------------------------------------------

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
>>> print len(events)
6161
>>> hist, bins = ae.loghist(event.maxima)
>>> plot( (bins[1:]+bins[:-1])/2, hist, "o")
```
![Graph](doc/hist.png)


[Documentation](http://jove1.github.io/ae/)
-------------------------------------------
