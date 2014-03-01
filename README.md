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
>>> print events.size
6161
>>> ae.hist(event.maxima)
```
![Graph](doc/hist.png)

