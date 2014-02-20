ae
==

Python acoustic emission tools.

Features
--------

* read *.wfs files
* event extractor
* interactive decimated waveform plotting
* optimized for files bigger than RAM

[Example](http://nbviewer.ipython.org/github/jove1/ae/blob/master/doc/example.ipynb)
------------------------------------------------------------------------------------

```python
>>> import ae
>>> f = ae.open("M5.wfs")
100% 0.65s
>>> print f.size
642798592
>>> f.plot()
100% 1.71s
```
![Graph](doc/view.png)
```python
>>> events = f.get_events(0.02)
100% 3.55s
6161 events
>>> hist, bins = ae.loghist([e.max for e in events])
>>> plot( (bins[1:]+bins[:-1])/2, hist, "o")
```
![Graph](doc/hist.png)


[Documentation](http://jove1.github.io/ae/)
-------------------------------------------
