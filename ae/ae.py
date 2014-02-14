#!/usr/bin/python

def open(fname):
    import os.path
    _, ext = os.path.splitext(fname)
    if ext.lower() == ".wfs":
        from .wfs import WFS
        return WFS(fname)
    else:
        raise NotImplementedError("Unknown format {}".format(ext))

class Event:
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
    def __init__(self, fname):
        raise NotImplementedError() 

    def plot(self, **kwargs):
        raise NotImplementedError() 
   
    def get_events(self, thresh, hdt=1000, dead=1000):
        raise NotImplementedError()
