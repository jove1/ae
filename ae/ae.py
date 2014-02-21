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

def count(data, thresh):
    return np.logical_and(data[:-1]<thresh, data[1:]>=thresh).sum()

from collections import namedtuple
Event = namedtuple("Event", "start, data")
Event.duration = property(lambda e: e.data.size )
Event.energy = property(lambda e: (e.data**2).sum() )
Event.max = property(lambda e: e.data.max() )
Event.rise_time = property(lambda e: np.argmax(e.data) )

Event.count = lambda e, thresh: count(e.data, thresh)

class Events(list):

    def __init__(self, source, thresh, pre, post):
        self.source = source
        self.thresh = thresh
        self.pre = pre
        self.post = post

    def __repr__(self):
        return "Events(<{} events>)".format(len(self))

    durations = property(lambda self: np.array([e.duration for e in self]))
    energies = property(lambda self: np.array([e.energy for e in self]))
    maxima = property(lambda self: np.array([e.max for e in self]))
    rise_times = property(lambda self: np.array([e.rise_time for e in self]))

    counts = property(lambda self: np.array([ e.count(self.thresh) for e in self]))

    def add_event(self, start, a, b, prev_data, data):
        pre = self.pre
        post = self.post
        assert a<b # sanity

        if a-pre < 0:
            assert a-pre >= -prev_data.size
            if b+post <0:
                # whole event in prev_data, we waited only for dead time 
                ev_data = prev_data[a-pre:b+post]*self.source.datascale
            else:
                # part in prev_data part in data
                assert b+post <= data.size
                ev_data = np.concatenate((
                    prev_data[a-pre:],
                    data.flat[:b+post]
                ))*self.source.datascale
        else:
            if b+post < data.size:
                # all in data
                ev_data = data.flat[a-pre:b+post]*self.source.datascale
            else:
                # pad with zeros
                assert a-pre <= data.size
                ev_data = np.concatenate((
                    data.flat[a-pre:],
                    np.zeros(b+post-data.size, dtype=self.dtype)
                ))*self.source.datascale
        
        assert ev_data.size == pre + b-a + post
        self.append( Event(start, ev_data) )


class Data:
    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.fname)

    progress = True
    def iter_blocks(self, start=0, stop=float('inf'), channel=slice(None), progress=None):
        if progress is None:
            progress = self.progress
        import sys, time
        start_time = time.time()
        for pos, raw in self.raw_iter_blocks(start, stop):
            self.check_block(pos, raw)
            yield pos, self.get_block_data(raw)[..., channel]
            if progress:
                sys.stderr.write("\r{:.0f}%".format(100.*pos/self.size))
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
        block = 0
        for _, d in self.iter_blocks(channel=channel):
            d.min(axis=-1, out=mins[block:block+d.shape[0]])
            d.max(axis=-1, out=maxs[block:block+d.shape[0]])
            block += d.shape[0]
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
            for pos, d in self.iter_blocks(start=a//s*s, stop=b//s*s, channel=channel, progress=False):
                aa = np.clip(a//s*s-pos, 0, d.size)
                bb = np.clip(b//s*s-pos, 0, d.size)
                blocks.append(d.flat[aa:bb])
            d = np.concatenate(blocks)
            d.shape = (d.size//s, s)
            mins = maxs = d

        x = np.empty(2*mins.shape[0])
        y = np.empty(2*mins.shape[0])
        x[::2] = x[1::2] = np.arange(a//s*s, b//s*s, s)*self.timescale

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

    def get_events(self, thresh, hdt=0.001, dead=0.001, pretrig=0.001, channel=0):
        raw_thresh = int(thresh/self.datascale)
        raw_hdt = int(hdt/self.timescale) 
        raw_pre = int(pretrig/self.timescale)
        raw_dead = int(dead/self.timescale)

        last = None

        events = Events(source=self, thresh=thresh, pre=raw_pre, post=raw_hdt)

        prev_data = np.zeros(raw_hdt+raw_dead+raw_pre, dtype=self.dtype)
        from .event_detector import process_block
        for pos, data in self.iter_blocks(channel=channel):
            ev, last = process_block(data, raw_thresh, hdt=raw_hdt, dead=raw_dead, event=last, pos=pos)
            for start,end in ev:
                events.add_event(start, start-pos, end-pos, prev_data, data)
            prev_data = data.flat[-(raw_hdt+raw_dead+raw_pre):]
        if last:
            start, end = last
            events.add_event(start, start-pos, end-pos, None, data)

        return events

class SDCF(Data):
    def __init__(self, fname, checks=False):
        import glob
        self.fname = fname 
        self.fnames = sorted(glob.glob(fname.replace("-000000.sdcf", "-*.sdcf")))
        self.checks = checks
       
        with file(fname,"rb") as fh:
            self.meta = fh.read(124)
        self.datascale = 1/32768.
        self.timescale = 1e-6
        self.timeunit = "?"
        self.dataunit = "?"

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

    def check_block(self, pos, raw):
        if self.checks:
            assert pos == raw['data']['offset'][0,0]

    def raw_iter_blocks(self, start=0, stop=float('inf')):
        import io, os

        offset = 0
        buffer = np.empty(1, self.block_dtype)
        block_size = self.get_block_data(buffer)[0,...,0].size
        
        pos = start//block_size*block_size
        seek = start//block_size*buffer.itemsize
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
                        yield pos, buffer
                        pos += buffer.size*block_size
                        if pos > stop:
                            return

        remains = offset // buffer.itemsize
        if remains:
            yield pos, buffer[:remains]

class WFS(Data):

    def __init__(self, fname, checks=False, unknown_meta=False):
        self.fname = fname 
        self.checks = checks

        import os
        with file(self.fname, "rb") as fh:
            file_size = os.fstat(fh.fileno()).st_size
            self._offset = self.parse_meta(fh.read(1024), unknown_meta=unknown_meta)
        self.calc_sizes(file_size-self._offset)

        self.datascale = self.meta['hwsetup']['max.volt']/32768.
        self.timescale = 0.001/self.meta['hwsetup']['rate']
        self.timeunit = "s"
        self.dataunit = "V"

    def parse_meta(self, data, unknown_meta=False):
        from struct import unpack_from,  calcsize
        from collections import OrderedDict
        self.meta = OrderedDict()
        offset = 0
        while offset < len(data):
            size, id1, id2 = unpack_from("<HBB", data, offset)
            if (size, id1, id2) == (2076, 174, 1):
                return offset
            offset += 2
            if id1 in (173, 174): 
                # these have two ids
                offset += 2
                size -= 2
                if (id1, id2) == (174,42):
                    fmt = [("ver", "H"),
                         ("AD", "B"),
                         ("num", "H"),
                         ("size", "H"),

                         ("id", "B"),
                         ("unk1", "H"),
                         ("rate", "H"),
                         ("trig.mode", "H"),
                         ("trig.src", "H"),
                         ("trig.delay", "h"),
                         ("unk2", "H"),
                         ("max.volt", "H"),
                         ("trig.thresh", "H"),
                    ]
                    sfmt = "<"+"".join(code for name,code in fmt)
                    assert calcsize(sfmt) == size
                    self.meta['hwsetup'] = OrderedDict(zip(
                        [name for name, code in fmt], 
                        unpack_from(sfmt, data, offset)))
                elif unknown_meta:
                    self.meta[(id1,id2)] = data[offset:offset+size]
            else: 
                # only one id
                offset += 1
                size -= 1
                if id1 == 99:
                    self.meta['date'] = data[offset:offset+size].rstrip("\0\n")
                elif id1 == 41:
                    self.meta['product'] = OrderedDict([
                        ("ver", unpack_from("<xH", data, offset)[0]), 
                        ("text", data[offset+3:offset+size].rstrip("\r\n\0\x1a"))])
                elif unknown_meta:
                    self.meta[id1] = data[offset:offset+size]
            offset += size
        raise ValueError("Data block not found")


    block_dtype = np.dtype([
            ("size", "u2"), 
            ("id1", "u1"),
            ("id2", "u1"),
            ("unknown", "V26"), 
            ("data", "i2", (1024,1))])
    get_block_data = staticmethod(lambda d: d['data'])
 
    def check_block(self, pos, raw):
        if self.checks:
            assert np.alltrue(raw['size'] == 2076) 
            assert np.alltrue(raw['id1'] == 174) 
            assert np.alltrue(raw['id2'] == 1) 

    def raw_iter_blocks(self, start=0, stop=float('inf')):
        import io, struct
        buffer = np.empty(8000, self.block_dtype)
        block_size = self.get_block_data(buffer)[0,...,0].size
        
        pos = start//block_size*block_size
        seek = start//block_size*buffer.itemsize
        with io.open(self.fname, "rb", buffering=0) as fh:
            fh.seek( self._offset + seek)
            while True:
                read = fh.readinto(buffer)
                if read < buffer.size*buffer.itemsize:
                    break
                else:
                    yield pos, buffer
                    pos += buffer.size*block_size
                    if pos > stop:
                        return 

        remains = read // buffer.itemsize
        if remains:
            yield pos, buffer[:remains]

def open(fname, checks=False):
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
        return WFS(fname, checks=checks)
    elif ext == ".sdcf":
        return SDCF(fname, checks=checks)
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
