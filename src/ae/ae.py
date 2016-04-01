#!/usr/bin/python 

def xpan(ax=None):
    """
    Activates matplotlib pan/zoom tool, constrained to x direction for current axes.
    """
    def drag_pan(self, button, key, x,y):
        return self.__class__.drag_pan(self, button, "x" if key is None else key, x,y)

    if ax is None:
        from matplotlib.pyplot import gca
        ax = gca()

    import types 
    ax.drag_pan = types.MethodType(drag_pan, ax)
    ax.figure.canvas.toolbar.pan()

try:
    from matplotlib.ticker import ScalarFormatter
except ImportError:
    pass
else:
    class TimeFormatter(ScalarFormatter):
        def format_data(self, value):
            'return a formatted string representation of a number'
            if self._useLocale:
                s = locale.format_string(self.format, (value,))
            else:
                s = self.format % value
            s = self._formatSciNotation(s)
            return self.fix_minus(s)

        def format_data_short(self,value):
            more = 1
            s = '%1.*f' % (int(self.format[3:-1])+more, value)
            #return s[:-more] + " " + s[-more:]
            return s

import numpy as np

def loghist(data, bins=50, range=None, density=None):
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
    hist, bins = np.histogram(data, bins, density=density)
    return hist, bins

def random_power(xmin, a, size=1):
    return xmin*(1-random.uniform(size=size))**(1./(a+1))

def bin_centers(bins):
    return (bins[1:] + bins[:-1])/2.

def join_bins(bins, counts, mincount=10):
    newbins, newcounts = [bins[0]], []
    s = 0
    for a,b in zip(counts,bins[1:]):
        s += a
        if s < mincount:
            continue
        newcounts.append(s)
        newbins.append(b)
        s = 0
    if s > 0:
        newcounts.append(s)
        newbins.append(b)
    return asarray(newbins), asarray(newcounts)

def cdf(data):
    return np.sort(data), np.arange(data.size,0,-1)

def mle(xmin, data):
    d = data[data>=xmin]
    a = 1 - data.size/sum(log(xmin/d))
    return -a, (a-1)/sqrt(d.size)       

def hist(data, bins=50, range=None, ax=None, density=None):
    if ax is None:
        from matplotlib.pyplot import gca
        ax = gca()

    hist, bins = loghist(data, bins=bins, range=range, density=density)
    l, = ax.plot( (bins[1:]+bins[:-1])/2, hist, "o")
    ax.loglog()
    ax.grid(True)
    return hist, bins, l

def count(data, thresh):
    return np.logical_and(data[:-1]<thresh, data[1:]>=thresh).sum()

from collections import namedtuple
Event = namedtuple("Event", "start, stop, data")
Event.duration = property(lambda e: e.data.size )
Event.energy = property(lambda e: (e.data**2).sum() )
Event.max = property(lambda e: e.data.max() )
Event.rise_time = property(lambda e: np.argmax(e.data) )
Event.count = lambda e, thresh: count(e.data, thresh)

from scipy.signal import welch
Event.psd = lambda e,**kwargs: welch(e.data,**kwargs)

class Events(np.ndarray):
    extra_attributes = ["source", "thresh", "pre", "hdt", "dead"]

    def __new__(cls, data, **kwargs):
        self = np.ndarray.__new__(cls, len(data), dtype=object)
        self[:] = data
        for a in self.extra_attributes: # initialize attributes
            setattr(self, a, kwargs.get(a))
        return self

    def __array_finalize__(self, obj):
        if obj is not None:
            for a in self.extra_attributes: # copy attributes
                setattr(self, a, getattr(obj, a, None))

    starts = property(lambda self: np.array([e.start for e in self])*self.source.timescale)
    durations = property(lambda self: np.array([e.duration for e in self])*self.source.timescale)
    energies = property(lambda self: np.array([e.energy for e in self])*self.source.timescale)
    maxima = property(lambda self: np.array([e.max for e in self]))
    rise_times = property(lambda self: np.array([e.rise_time for e in self])*self.source.timescale)

    counts = property(lambda self: np.array([ e.count(self.thresh) for e in self]))
    
    def psd(self,**kwargs):
        Pxxs = np.array([e.psd(**kwargs)[1] for e in self])
        f = e.psd(**kwargs)[0]
        return f,  Pxxs

    def plot(self, ax=None):
        from itertools import izip, repeat
        if ax is None:
            from matplotlib.pyplot import gca
            axiter = repeat( gca() )
        else:
            try:
                axiter = iter(ax)
            except TypeError:
                axiter = repeat(ax)
        s = self.source.timescale
        axes = set()
        for ax,e in izip(axiter, self):
            ax.plot(s*(np.arange(e.data.size)+e.start-self.pre), e.data, c="b")
            ax.axvspan(s*(e.start-self.pre), s*e.start,  color="g", alpha=0.25)
            ax.axvspan(s*e.start, s*e.stop,  color="g", alpha=0.5)
            ax.axvspan(s*e.stop, s*(e.stop+self.hdt),  color="g", alpha=0.25)
            ax.axvspan(s*(e.stop+self.hdt), s*(e.stop+self.hdt+self.dead),  color="r", alpha=0.25)
            axes.add(ax)

        for ax in axes:
            ax.xaxis.set_major_formatter(TimeFormatter())
            ax.axhline(0, c="k")
            ax.axhline(self.thresh, c="k", ls="--")

        return axes

def text_progress(percent, time):
    import sys
    sys.stderr.write("\r")
    try:
        from IPython.display import clear_output
    except ImportError:
        pass
    else:
        clear_output(stdout=False, other=False)

    if percent < 100:
        sys.stderr.write("{:.0f}%".format(percent))
    else:
        sys.stderr.write("{:.0f}% {:.2f}s\n".format(percent, time))
    sys.stderr.flush()

def no_progress(percent, time):
    pass

class Data:
    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.fname)
    
    progress = staticmethod(text_progress)
    def iter_blocks(self, start=0, stop=float('inf'), channel=slice(None), progress=None):
        if progress is None:
            progress = self.progress

        import sys, time
        start_time = time.time()
        for pos, raw in self.raw_iter_blocks(start, stop):
            self.check_block(pos, raw)
            yield pos, self.get_block_data(raw)[..., channel]
            progress(100.*pos/self.size, time.time()-start_time)
        
        progress(100, time.time()-start_time)

    def calc_sizes(self, file_size):
        n_blocks = file_size // self.block_dtype.itemsize
        rest = file_size % self.block_dtype.itemsize
        if rest:
            import warnings
            warnings.warn("{} bytes at the end of file won't fit into blocks".format(rest))

        tmp = self.get_block_data( np.empty(0, self.block_dtype) )
        self.dtype = tmp.dtype
        self.shape = (n_blocks,) + tmp.shape[1:-1]
        self.size = np.prod(self.shape)

        self.channels = tmp.shape[-1]
        assert self.channels == len(self.datascale)

    def get_min_max(self, channel=0):
        try:
            return self._min_max_cache[channel] 
        except AttributeError:
            self._min_max_cache = {}
        except KeyError:
            pass

        cachefn = self.fname+".envelope.cache.{}.npz".format(channel)
        try:
            d = np.load(cachefn)
            mins, maxs = d['mins'], d['maxs']
            self._min_max_cache[channel] = (mins,maxs)
            #print "envelope in cache"
            return (mins,maxs)
        except:
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
        np.savez(cachefn, mins=mins, maxs=maxs)

        return mins, maxs

    def resample(self, range, channel=0, num=768):
        from math import floor, ceil
        def clip(x,a,b):
            return min(max(x,a),b)
        a,b = range
        a = int(floor(a/self.timescale))
        b = int(ceil(b/self.timescale)) + 1
        s = max((b-a)//num, 1)
        #print "resample", s, b-a

        a = clip(a, 0, self.size)
        b = clip(b, 0, self.size)

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
            for pos, d in self.iter_blocks(start=a//s*s, stop=b//s*s, channel=channel, progress=no_progress):
                aa = clip(a//s*s-pos, 0, d.size)
                bb = clip(b//s*s-pos, 0, d.size)
                blocks.append(d.flat[aa:bb])
            d = np.concatenate(blocks)
            d.shape = (d.size//s, s)
            mins = maxs = d

        x = np.empty(2*mins.shape[0])
        y = np.empty(2*mins.shape[0])
        x[::2] = x[1::2] = np.arange(a//s*s, b//s*s, s)*self.timescale

        mins.min(axis=-1, out=y[::2])
        maxs.max(axis=-1, out=y[1::2])
        y *= self.datascale[channel]
        return x,y

    def plot(self, channel=0, ax=None, **kwargs):
        if ax is None:
            from matplotlib.pyplot import gca
            ax = gca()
        line, = ax.plot([], [], **kwargs)

        def update(ax):
            try:
                x, y = self.resample(ax.viewLim.intervalx, channel=channel)
                line.set_data(x,y)
                ax.figure.canvas.draw_idle()
            except:
                import traceback
                traceback.print_exc()

        ax.callbacks.connect('xlim_changed', update)
        ax.set_xlim(0,(self.size-1)*self.timescale)
        ax.relim()
        ax.autoscale_view(scalex=False)
        ax.xaxis.set_major_formatter(TimeFormatter())

        return line


    def get_events(self, thresh, hdt=0.001, dead=0.001, pretrig=0.001, channel=0, limit=0):
        raw_thresh = int(thresh/self.datascale[channel])
        raw_hdt = int(hdt/self.timescale) 
        raw_pre = int(pretrig/self.timescale)
        raw_dead = int(dead/self.timescale)
        raw_limit = int(limit/self.timescale)


        def _get_event(start, stop, pos, prev_data, data):
            a = start-raw_pre-pos
            b = stop+raw_hdt-pos
            datascale = self.datascale[channel]

            assert a<b, (a,b) # sanity

            if a < 0:
                assert a >= -prev_data.size, (a, prev_data.size)
                if b <0:
                    # whole event in prev_data, we waited only for dead time 
                    ev_data = prev_data[a:b]*datascale
                else:
                    # part in prev_data part in data
                    assert b <= data.size, (b, data.size)
                    ev_data = np.concatenate((
                        prev_data[a:],
                        data.flat[:b]
                    ))*datascale
            else:
                if b < data.size:
                    # all in data
                    ev_data = data.flat[a:b]*datascale
                else:
                    # pad with zeros
                    assert a <= data.size, (a, data.size)
                    ev_data = np.concatenate((
                        data.flat[a:],
                        np.zeros(b-data.size, dtype=data.dtype)
                    ))*datascale
            
            assert ev_data.size == raw_pre + stop-start + raw_hdt
            return Event(start, stop, ev_data)

        def _add_event(*args):
            try:
                events.append(_get_event(*args))
            except:
                import traceback
                traceback.print_exc()

        last = None
        events = []
        prev_data = np.zeros(raw_pre, dtype=self.dtype)
        from .event_detector import process_block
        for pos, data in self.iter_blocks(channel=channel):
            ev, last = process_block(data.astype("i2"), raw_thresh, hdt=raw_hdt, dead=raw_dead, event=last, pos=pos, limit=raw_limit)
            for start,stop in ev:
                _add_event(start, stop, pos, prev_data, data)
            start = last[0] - pos if last else 0
            prev_data = data.flat[start-raw_pre:]
        if last:
            _add_event(last[0], last[1], pos, None, data)

        return Events(source=self, thresh=thresh, pre=raw_pre, hdt=raw_hdt, dead=raw_dead, data=events)

    def save_bdat(self, fname, channel=0, range=(0,float("inf"))):
        """
        save data to fname as bdat file format 
        channel and data range (in samples) to save can be specfied
        """

        from __builtin__ import open
        start, end = range
        with open(fname, "wb") as fh:
            fh.write("\0"*12)
            for pos, data in self.iter_blocks(start=start, stop=end, channel=channel):
                if pos < start or pos+data.size > end:
                    a = max(0, start-pos)
                    b = min(data.size, end-pos)
                    data.flat[a:b].astype(">i2").tofile(fh)
                else:
                    data.astype(">i2").tofile(fh)
        
    def save_wav(self, fname, channel=0, range=(0,float("inf")), rate=None):
        """
        save data to fname as wav file format 
        channel, data range (in samples) to save can be specfied
        if rate is not None it overides the sample rate saved into wav
        """

        import wave
        
        w = wave.open(fname,"w")
        w.setnchannels(1)
        w.setsampwidth(self.dtype.itemsize)
        if rate is None:
            w.setframerate(1./self.timescale)
        else:    
            w.setframerate(rate)

        start, end = range
        for pos, data in self.iter_blocks(start=start, stop=end, channel=channel):
            if pos < start or pos+data.size > end:
                a = max(0, start-pos)
                b = min(data.size, end-pos)
                w.writeframes(data.flat[a:b].astype("<i2").tostring())
            else:
                w.writeframes(data.astype("<i2").tostring())
        w.close()


from collections import OrderedDict
class PrettyOrderedDict(OrderedDict):
    def __str__(d, prefix=""):
        indent = "    "
        s = ["OrderedDict("]
        for k,v in d.items():
            if isinstance(v, PrettyOrderedDict):
                s.append(prefix+indent+"({!r}, {}),".format(k, v.__str__(prefix+indent)))
            else:
                s.append(prefix+indent+"({!r}, {!r}),".format(k,v))
        s.append(prefix+")")
        return "\n".join(s)

class SDCF(Data):
    def __init__(self, fname, **kwargs):
        import glob
        self.fname = fname 
        self.fnames = sorted(glob.glob(fname.replace("-000000.sdcf", "-*.sdcf")))
        self.checks = kwargs.get("checks", False)
    
        self.parse_meta(fname, unknown_meta=kwargs.get("unknown_meta", False))


        
        self.datascale = [1/32768.*2*10**((-35-gain)/20.) for gain in self.meta['2']['gains'][1:]]
        self.timescale = 1./self.meta['2']['rate']
        self.timeunit = "s"
        self.dataunit = "V"

        import os
        file_size = sum( os.stat(fname).st_size for fname in self.fnames)
        self.calc_sizes(file_size)

    def parse_meta(self, fname, unknown_meta=False):
        with file(fname,"rb") as fh: # np.rec.fromfile would not handle unicode filename
            meta = np.rec.fromfile(fh, dtype=[ ("meta1", self.meta1_dtype), ("meta2", self.meta2_dtype)], shape=())
        
        self.meta = PrettyOrderedDict()
        self.meta['1'] = PrettyOrderedDict( zip(meta.meta1.dtype.names, meta.meta1.tolist()) )
        self.meta['2'] = PrettyOrderedDict( zip(meta.meta2.dtype.names, meta.meta2.tolist()) )
        
        assert self.meta['1']['magic'] == 0xfbdffbdf
        assert self.meta['2']['magic'] == 0xebd7ebd7
        self.meta['1']['description'] = self.meta['1']['description'].rstrip("\0 ")
        self.meta['1']['creator'] = tuple(self.meta['1']['creator'])
        import datetime
        self.meta['1']['timestamp'] = datetime.datetime.fromtimestamp(self.meta['1']['timestamp']*1e-6)
        self.meta['2']['gains'] = tuple(self.meta['2']['gains'])
       
        if not unknown_meta:
            del self.meta["1"]["magic"]
            del self.meta["1"]["unk1"]
            del self.meta["1"]["unk2"]
            del self.meta["1"]["checksum"]

            del self.meta["2"]["magic"]
            del self.meta["2"]["unk1"]
            del self.meta["2"]["unk2"]
            del self.meta["2"]["unk3"]
            del self.meta["2"]["checksum"]
        else:
            self.meta["1"]["unk1"] = str(self.meta["1"]["unk1"])
            self.meta["1"]["unk2"] = str(self.meta["1"]["unk2"])
            
            self.meta["2"]["unk1"] = str(self.meta["2"]["unk1"])
            self.meta["2"]["unk2"] = str(self.meta["2"]["unk2"])
            self.meta["2"]["unk3"] = str(self.meta["2"]["unk3"])


    meta1_dtype = np.dtype([
             ("magic", "u4"),
             ("size", "i2"), ("unk1","V10"),
             #("unk1", "V12"),
             ("description", "S32"),
             ("timestamp", "u8"),
             ("creator", [("name","S4"),("version","u8")]),
             ("unk2", "V2"),
             ("checksum", "u4"),
    ]) 

    meta2_dtype = np.dtype([
             ("magic", "u4"),
             ("size", "i2"), ("unk1","V8"),
             #("unk1", "V10"),
             ("rate", "u4"),
             ("unk2", "V6"),
             ("gains", "f4", (5,)),
             ("unk3", "V2"),
             ("checksum", "u4"),
    ])

    data_dtype = np.dtype([
             ("magic", "u4"),
             ("size1", "i4"),
             ("one", "i4"),
             ("offset", "i4"),
             ("zero", "i4"),
             ("size2", "i4"),
             #arbitrary split for caching min,max values, when plotting
             #("data", ">i2", (16380,4)), 
             ("data", ">i2", (15,1092,4)), 
             ("checksum", "u4"),
    ])

    block_dtype = np.dtype([
             ("meta1", meta1_dtype ),
             ("meta2", meta2_dtype ),
             ("data", data_dtype, (129,) ),
    ])
    get_block_data = staticmethod(lambda d: d['data']['data'])

    def check_block(self, pos, raw):
        if self.checks:
            assert np.alltrue(raw['meta1']['magic'] == 0xfbdffbdf)
            assert np.alltrue(raw['meta2']['magic'] == 0xebd7ebd7)
            assert np.alltrue(raw['data']['magic'] == 0xe5a7e5a7)
            assert np.alltrue(raw['data']['size1'] == 131060)
            assert np.alltrue(raw['data']['size2'] == 131040)
            assert np.alltrue(raw['data']['one'] == 1)
            assert np.alltrue(raw['data']['zero'] == 0)
            assert pos == raw['data']['offset'][0,0]

            offsets = np.arange(raw['data'].size)
            offsets.shape = raw['data'].shape
            assert np.alltrue( raw['data']['offset'] == offsets*16380 + pos)

    def raw_iter_blocks(self, start=0, stop=float('inf')):
        import io, os

        offset = 0
        buffer = np.empty(1, self.block_dtype)
        block_size = self.get_block_data(buffer)[0,...,0].size
        
        pos = start//block_size*long(block_size)
        seek = start//block_size*long(buffer.itemsize)
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
        rest = offset % buffer.itemsize
        if remains:
            yield pos, buffer[:remains]
        if rest:
            import warnings
            warnings.warn("{} bytes left in the buffer".format(rest))


class ContigousBase(Data):

    get_block_data = staticmethod(lambda d: d['data'])
 
    def check_block(self, pos, raw):
        pass
    
    def raw_iter_blocks(self, start=0, stop=float('inf')):
        import io, struct
        buffer = np.empty(8192, self.block_dtype)
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
        rest = read % buffer.itemsize
        if remains:
            yield pos, buffer[:remains]
        if rest:
            import warnings
            warnings.warn("{} bytes left in the buffer".format(rest))

class BDAT(ContigousBase):
    def __init__(self, fname, **kwargs):
        self.fname = fname 
        self.checks = kwargs.get("checks", False)

        import os
        with file(self.fname, "rb") as fh:
            file_size = os.fstat(fh.fileno()).st_size
            self._offset = 12

        gain = kwargs.get("gain", 0)
        self.datascale = [1/32768.*2*10**((-35-gain)/20.)]
        self.timescale = 1e-6
        self.timeunit = "?"
        self.dataunit = "?"
        self.meta = None

        self.calc_sizes(file_size-self._offset)

    block_dtype = np.dtype([
            ("data", ">i2", (1024,1))])


class WAV(ContigousBase):

    def __init__(self, fname, **kwargs):
        self.fname = fname 
        self.checks = kwargs.get("checks", False)

        import os
        with file(self.fname, "rb") as fh:
            file_size = os.fstat(fh.fileno()).st_size
            self._offset = self.parse_meta(fh.read(1024))
        
        assert self.meta['fmt'] == 1

        self.block_dtype = np.dtype([
            ("data", "<i{}".format(self.meta['bps']/8), (1024,self.meta['nchan']))])

        self.datascale = [1.]*self.meta['nchan']
        self.timescale = 1./self.meta['rate']
        self.timeunit = "s"
        self.dataunit = "?"

        self.calc_sizes(file_size-self._offset)

    def parse_meta(self, data):
        from struct import unpack_from,  calcsize
        self.meta = PrettyOrderedDict()
        offset = 0
        while offset < len(data):
            chunk, size = unpack_from("<4sL", data, offset)
            if chunk == "RIFF":
                waveid, = unpack_from("<4s", data, offset+8)
                assert waveid == "WAVE"
                offset += 12 # "dive" into this chunk, don't use size to skip over
            
            elif chunk == "fmt ":
                assert size ==  16
                fmt, nchan, rate, _, _, bps = unpack_from("<HHLLHH", data, offset+8)
                self.meta['fmt'] = fmt
                self.meta['nchan'] = nchan
                self.meta['rate'] = rate
                self.meta['bps'] = bps
                
                offset += 8 + size

            elif chunk == "data":
                return offset + 8

            else:
                import warnings
                warnings.warn("unknown chunk {!r}".format(chunk))
                offset += 8 + size
            
        raise ValueError("Data block not found")


class WFS(Data):

    def __init__(self, fname, **kwargs):
        self.fname = fname 
        self.checks = kwargs.get("checks", False)

        import os
        with file(self.fname, "rb") as fh:
            file_size = os.fstat(fh.fileno()).st_size
            self._offset = self.parse_meta(fh.read(1024), unknown_meta=kwargs.get("unknown_meta", False))

        # determine number of channels
        import io
        buf = np.empty(10, self.ch_block_dtype)
        with io.open(self.fname, "rb", buffering=0) as fh:
            fh.seek( self._offset)
            fh.readinto(buf)
        for ch in xrange(1,10):
            if buf['chan'][ch] == ch +1:
                continue
            elif buf['chan'][ch] == 1:
                break
            else:
                raise ValueError("Invalid channels: %s".format(buf['chan']))

        self.block_dtype = np.dtype([ ('ch_data', self.ch_block_dtype, (ch,))])

        self.datascale = [self.meta['hwsetup']['max.volt']/32768.]*ch
        self.timescale = 0.001/self.meta['hwsetup']['rate']
        self.timeunit = "s"
        self.dataunit = "V"

        self.calc_sizes(file_size-self._offset)

    def parse_meta(self, data, unknown_meta=False):
        from struct import unpack_from,  calcsize
        self.meta = PrettyOrderedDict()
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
                    self.meta['hwsetup'] = PrettyOrderedDict(zip(
                        [name for name, code in fmt], 
                        unpack_from(sfmt, data, offset)))
                    if self.meta['hwsetup']['AD'] == 2:
                        self.meta['hwsetup']['AD'] = "16-bit signed"
                elif unknown_meta:
                    self.meta[(id1,id2)] = data[offset:offset+size]
            else: 
                # only one id
                offset += 1
                size -= 1
                if id1 == 99:
                    self.meta['date'] = data[offset:offset+size].rstrip("\0\n")
                elif id1 == 41:
                    self.meta['product'] = PrettyOrderedDict([
                        ("ver", unpack_from("<xH", data, offset)[0]), 
                        ("text", data[offset+3:offset+size].rstrip("\r\n\0\x1a"))])
                elif unknown_meta:
                    self.meta[id1] = data[offset:offset+size]
            offset += size
        raise ValueError("Data block not found")


    ch_block_dtype = np.dtype([
            ("size", "u2"),
            ("id1", "u1"),
            ("id2", "u1"),
            ("unknown1", "S6"),
            ("chan", "u1"),
            ("unknown2", "S19"),
            ("data", "i2", (1024))])

    get_block_data = staticmethod(lambda d: d['ch_data']['data'].swapaxes(-1,-2))

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
        rest = read % buffer.itemsize
        if remains:
            yield pos, buffer[:remains]
        if rest:
            if rest == 9 and np.alltrue(buffer.view('B')[read-rest:read] == (7, 0, 15, 255, 255, 255, 255, 255, 127)):
                pass
            else:
                import warnings
                warnings.warn("{} bytes left in the buffer".format(rest))

def open(fname=None, **kwargs):
    """
    Opens file using appropriate :class:`Data` subclass. Currently 
    supports .wfs format. 

    :param str fname: Name of file to open
    :return: :class:`.ae.Data` subclass instance
    """
    
    if fname is None:
        filetypes = [('WFS', '.wfs'),
                     ('SDCF', '*-000000.sdcf'),
                     ('BDAT', '.bdat'),
                     ('WAV', '.wav'),
                     ]
        anytype = ('Any AE file', ' '.join(pattern for name, pattern in filetypes))

        import os
        if os.name == "nt":
            # on windows the last item gets selected
            filetypes.append(anytype)
        else:
            filetypes.insert(0, anytype)
        
        try:
            kw = {'parent': kwargs.pop('parent') } 
        except KeyError:
            kw = {}
        import tkFileDialog
        fname = tkFileDialog.askopenfilename(filetypes=filetypes, **kw)
        
        if not fname:
            raise ValueError

    import os.path
    _, ext = os.path.splitext(fname)
    ext = ext.lower()
    if ext == ".wfs":
        return WFS(fname, **kwargs)
    elif ext == ".sdcf":
        return SDCF(fname, **kwargs)
    elif ext == ".bdat":
        return BDAT(fname, **kwargs)
    elif ext == ".wav":
        return WAV(fname, **kwargs)
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
        for ch in range(x.channels):
            x.plot(channel=ch, label="ch#{}".format(ch))
        #xpan()
        legend()
        grid()
    show()
