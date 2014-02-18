#!/usr/bin/python 

def xpan():
    """
    Activates matplotlib pan/zoom tool, constrained to x direction for current axes.
    """
    def drag_pan(self, button, key, x,y):
        return self.__class__.drag_pan(self, button, "x" if key is None else key, x,y)
    from matplotlib.pyplot import gca
    import types 
    ax = gca()
    ax.drag_pan = types.MethodType(drag_pan, ax)
    ax.figure.canvas.toolbar.pan()

import numpy as np

def loghist(data, bins=50, range=None):
    """
    Creates logarithmically spaced bins and calls :func:`numpy.histogram`.

    :param ndarray data:
    :param int bins: numer of bins
    :param tuple range: histogram range, by default determined from minimim and maximum in data
    :return: (hist, bins) - histogram counts and bin boundaries 
    """
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

class Data:

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.fname)

    def iter_blocks(self, start=0, stop=float('inf'), channel=slice(None), progress=True):
        import sys, time
        start_time = time.time()
        for block, pos, raw in self.raw_iter_blocks(start, stop):
            self.check_block(block, pos, raw)
            yield block, pos, self.get_block_data(raw)[..., channel]
            if progress:
                sys.stderr.write("\r{:.0f}%".format(100.*block/self.shape[0]))
                sys.stderr.flush()
        if progress:
            sys.stderr.write("\r{:.0f}% {:.2f}s\n".format(100., time.time()-start_time))

    def calc_sizes(self, file_size):
        n_blocks = file_size // self.block_dtype.itemsize
        print "rest", file_size % self.block_dtype.itemsize

        tmp = self.get_block_data( np.empty(0, self.block_dtype) )
        self.dtype = tmp.dtype
        self.shape = (n_blocks,) + tmp.shape[1:-1]
        self.size = np.prod(self.shape)

        self.channels = tmp.shape[-1]

    def get_min_max(self, channel=0):
        try:
            return self._min_max_cache[channel] 
        except AttributeError:
            self._min_max_cache = {}
        except KeyError:
            pass
        
        mins = np.empty(self.shape[:-1], dtype=self.dtype)
        maxs = np.empty(self.shape[:-1], dtype=self.dtype)
        for block, _, d in self.iter_blocks(channel=channel):
            d.min(axis=-1, out=mins[block:block+d.shape[0]])
            d.max(axis=-1, out=maxs[block:block+d.shape[0]])
        mins.shape = (mins.size,)
        maxs.shape = (maxs.size,)

        self._min_max_cache[channel] = (mins,maxs)
        return mins, maxs

    def resample(self, range, channel=0, num=768):
        a,b = range
        a = int(np.floor(a/self.timescale))
        b = int(np.ceil(b/self.timescale)) + 1
        s = max((b-a)//num, 1)
        #print "resample", s, b-a

        a = np.clip(a, 0, self.size)
        b = np.clip(b, 0, self.size)

        r = self.shape[-1]
        if s > r:
            s //= r
            a //= r
            b //= r
            mins, maxs = self.get_min_max(channel)
            mins = mins[a//s*s:b//s*s]
            mins.shape = (b//s-a//s, s)
            maxs = maxs[a//s*s:b//s*s]
            maxs.shape = (b//s-a//s, s)
            s *= r
            a *= r
            b *= r
        else:
            blocks = []
            for _, pos, d in self.iter_blocks(start=a//s*s, stop=b//s*s, channel=channel, progress=False):
                aa = np.clip(a//s*s-pos, 0, d.size)
                bb = np.clip(b//s*s-pos, 0, d.size)
                blocks.append(d.flat[aa:bb])
            d = np.concatenate(blocks)
            d.shape = (d.size//s, s)
            mins = maxs = d

        x = np.empty(2*mins.shape[0])
        y = np.empty(2*mins.shape[0])
        x[::2] = x[1::2] = np.arange(a//s*s*self.timescale,
                                     b//s*s*self.timescale,
                                     s*self.timescale)
        mins.min(axis=-1, out=y[::2])
        maxs.max(axis=-1, out=y[1::2])
        y *= self.datascale
        return x,y

    def plot(self, channel=0, **kwargs):
        from matplotlib.pyplot import gca
        ax = gca()
        line, = ax.plot([], [], **kwargs)

        def update(ax):
            x, y = self.resample(ax.viewLim.intervalx, channel=channel)
            line.set_data(x,y)
            ax.figure.canvas.draw_idle()
        
        ax.callbacks.connect('xlim_changed', update)
        ax.set_xlim(0,(self.size-1)*self.timescale)
        ax.relim()
        ax.autoscale_view(scalex=False)
        
        return line

class SDCF(Data):
    def __init__(self, fname, checks=False):
        import glob
        self.fname = fname 
        self.fnames = sorted(glob.glob(fname.replace("-000000.sdcf", "-*.sdcf")))
        self.checks = checks
       
        with file(fname,"rb") as fh:
            self.meta = fh.read(124)
        self.datascale = 1
        self.timescale = 1

        import os
        file_size = sum( os.stat(fname).st_size for fname in self.fnames)
        self.calc_sizes(file_size)

    block_dtype = np.dtype([
                ("magic", "S4"),
                ("unknown", "V120"),
                ("data", [
                    ("magic", "S4"),
                    ("size1", "i4"),
                    ("one", "i4"),
                    ("offset", "i4"),
                    ("zero", "i4"),
                    ("size2", "i4"),
                    #("data", ">i2", (16380,4)), 
                    #arbitrary split for caching min,max values, when plotting
                    ("data", ">i2", (15,1092,4)), 
                    ("checksum", "i4"),
                    ], (129,))
                ])
    get_block_data = staticmethod(lambda d: d['data']['data'])

    def check_block(self, block, pos, raw):
        if self.checks:
            assert pos == raw['data']['offset'][0,0]

    def raw_iter_blocks(self, start=0, stop=float('inf')):
        import io, os

        offset = 0
        buffer = np.empty(1, self.block_dtype)
        block_size = self.get_block_data(buffer)[0,...,0].size
        
        pos = start // block_size 
        
        seek = pos*buffer.itemsize
        for fname in self.fnames:
            with io.open(fname, "rb", buffering=0) as fh:
                file_size = os.fstat(fh.fileno()).st_size
                if seek < file_size:
                    fh.seek(seek)
                    seek = 0
                else:
                    seek -= file_size
                    continue

                while True:
                    view = buffer.view('B')[offset:]
                    read = fh.readinto(view)
                    if read < len(view):
                        offset += read
                        break
                    else:
                        assert offset + read == buffer.size*buffer.itemsize
                        offset = 0
                        yield pos, pos*block_size, buffer
                        pos += buffer.size
                        if pos*block_size > stop:
                            return

        remains = offset // buffer.itemsize
        if remains:
            yield pos, pos*block_size, buffer[:remains]

class WFS(Data):

    def __init__(self, fname, checks=False):
        self.fname = fname 
        self.datascale = 1
        self.timescale = 1
        self.checks = checks

        import struct, os, io
        signature = struct.pack("HBB", 2076, 174, 1)
        
        with file(self.fname, "rb") as fh:
            self._offset = fh.read(1024).find(signature)
            file_size = os.fstat(fh.fileno()).st_size
            self.meta = ""

        self.calc_sizes(file_size-self._offset)


    block_dtype = np.dtype([
            ("size", "u2"), 
            ("id1", "u1"),
            ("id2", "u1"),
            ("unknown", "V26"), 
            ("data", "i2", (1024,1))])
    get_block_data = staticmethod(lambda d: d['data'])
 
    def check_block(self, block, pos, raw):
        if self.checks:
            assert np.alltrue(raw['size'] == 2076) 
            assert np.alltrue(raw['id1'] == 174) 
            assert np.alltrue(raw['id2'] == 1) 

    def raw_iter_blocks(self, start=0, stop=float('inf')):
        import io, struct
        buffer = np.empty(8000, self.block_dtype)
        block_size = self.get_block_data(buffer)[0,...,0].size
        
        pos = start // block_size 

        with io.open(self.fname, "rb", buffering=0) as fh:
            fh.seek( self._offset + pos*buffer.itemsize )
            while True:
                read = fh.readinto(buffer)
                if read < buffer.size*buffer.itemsize:
                    break
                else:
                    yield pos, pos*block_size, buffer
                    pos += buffer.size
                    if pos*block_size > stop:
                        return 

        remains = read // buffer.itemsize
        if remains:
            yield pos, pos*block_size, buffer[:remains]

def open(fname):
    """
    Opens file using appropriate :class:`Data` subclass. Currently 
    supports .wfs format. 

    :param str fname: Name of file to open
    :return: :class:`.ae.Data` subclass instance
    """

    import os.path
    _, ext = os.path.splitext(fname)
    ext = ext.lower()
    if ext == ".wfs":
        return WFS(fname)
    elif ext == ".sdcf":
        return SDCF(fname)
    else:
        raise NotImplementedError("Unknown format {}".format(ext))

if __name__ == "__main__":
    from pylab import show, figure, grid, subplots_adjust, legend
    import event_detector

    for fname in [
            "data-0000001391607156071432-000000.sdcf",
            "M5.wfs",
            "original.wfs",
        ]:

        x = open(fname)
        print x
        print x.size, x.shape, x.dtype
        print repr(x.meta)

        if 0:
            l = []
            def f(a,b):
                l.append((a,b))
            det = event_detector.EventDetector(f)
            for o, d in x.iter_blocks(channel=0):
                det.process(d, 65)
            print len(l), "events"

        figure(figsize=(8,4))
        subplots_adjust(0.10,0.10,0.98,0.95)
        for ch in range(x.channels)[::-1]:
            x.plot(channel=ch, label="ch#{}".format(ch))
        xpan()
        legend()
        grid()
    show()
