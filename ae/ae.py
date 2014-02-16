#!/usr/bin/python

def open(fname):
    """
    Opens file using appropriate :class:`.ae.Data` subclass. Currently 
    supports .wfs format. 

    :param str fname: Name of file to open
    :return: :class:`.ae.Data` subclass instance
    """
    import os.path
    _, ext = os.path.splitext(fname)
    if ext.lower() == ".wfs":
        from .wfs import WFS
        return WFS(fname)
    else:
        raise NotImplementedError("Unknown format {}".format(ext))

def loghist(data, bins=50, range=None):
    """
    Creates logarithmically spaced bins and calls :func:`numpy.histogram`.

    :param ndarray data:
    :param int bins: numer of bins
    :param tuple range: histogram range, by default determined from minimim and maximum in data
    :return: (hist, bins) - histogram counts and bin boundaries 
    """
    import numpy as np
    data = np.asarray(data)
    if range is None:
        a,b = data.min(), data.max()
        if a == 0:
            a = b/1000.
    else:
        a,b = range
    bins = np.exp(np.linspace(np.log(a), np.log(b), bins))
    hist, bins = np.histogram(data, bins)
    return hist, bins


class Event:
    """
    Holds AE event data. 

    :ivar ndarray data: Event waveform
    :ivar float start:
    :ivar float end:
    :ivar float duration:
    :ivar float max:
    :ivar float max2:
    :ivar float energy:
    :ivar float rise:
    :ivar float count:
    """

    def __init__(self, start, end, data, thresh):
        self.start = start
        self.end = end
        self.data = data 

        self.duration = end-start
        self.energy = (data**2).sum()
        self.max = data.max()
        self.max2 = self.max**2
        from numpy import argmax
        self.rise = argmax(data)
        self.count = (data>thresh).sum()

class Data:
    """
    Holds AE waveform data. 
    
    :param str fname: Filename

    :ivar str fname: Filename
    :ivar ndarray data: Waveform in raw units
    :ivar float datascale: Multiply data by this to get volts
    :ivar float timescale: Sample spacing in seconds
    """

    def __init__(self, fname):
        """
        :param str fname: Name of file to open
        """
        raise NotImplementedError() 

    def plot(self, **kwargs):
        """
        :keyword: Passed to :class:`.plot.Downsampler`
        :return: :class:`.plot.Downsampler` instance
        """
        raise NotImplementedError() 
   
    def get_events(self, thresh, hdt=1000, dead=1000):
        """
        :param float thresh: Threshold
        :param float hdt: Hit definition time
        :param float dead: Dead time
        :return: list of :class:`.ae.Event` instaces
        """
        raise NotImplementedError()
