#!/usr/bin/python

def unpack(fields, buffer, offset=0):
    from struct import unpack_from, calcsize
    ret = []
    for fmt in fields:
        val = unpack_from(fmt, buffer, offset)
        if len(val) == 1:
            val = val[0]
        ret.append(val)
        offset += calcsize(fmt)
    ret.append(offset)
    return ret

from collections import OrderedDict
class Msg(OrderedDict):
    def __init__(self, id, data):
        OrderedDict.__init__(self)
        self.id = id

        self.label, func = self._messages[id]
        self.update( func(data) )

    def __str__(self):
        r = ["[{}: {}]".format(self.id, self.label)]
        for k,v in self.items():
            rv = repr(v)
            if k == "data" and len(rv) > 50:
                r.append("  {:<6} = {}...{}".format(k,rv[:50],len(v)) )
            else:
                r.append("  {:<6} = {}".format(k,rv))
        return "\n".join(r)


    def _msg_174_106(data):
        l = map(ord, data)
        assert len(l) == l[1] + 2
        return [ ("group", l[0]), 
                 ("channels", l[2:])]
    
    def _msg_174_1(data):
        ver, sync, chan, fifo_w1, fifo_w2, fifo_r1, fifo_r2, o = unpack("HLLLLLL", data)
        import array
        d = array.array("h")
        d.fromstring(data[o:])
        return [("ver", ver),
                ("sync", sync),
                ("chan",chan),
                ("fifo_w", (fifo_w1,fifo_w2*4)),
                ("fifo_r", (fifo_r1,fifo_r2*4)),
                ("data",d)]

    def _msg_173_1(data):
        time1, time2, chan, o = unpack("LHH", data)
        import array
        d = array.array("h")
        d.fromstring(data[o:])
        return [("time", (time1,time2)),
                ("chan",chan),
                ("data", d)]
 
    def _msg_174_174(data):
        ver, n, o = unpack("HH", data)
        conf = []
        for i in range(n):
            chan, t1, t2, off, o = unpack("BLHQ", data, o)
            conf.append( [("chan",chan), ("t",(t1,t2)) , ("off",off*2)])
        return [("ver", ver),
                ("conf", conf)]

    def _msg_174_42(data):
        ver, ad_type, num, size, o = unpack("HBHH", data)
        conf = []
        while o < len(data):
            r = unpack("BHHHHhHHH", data, o)
            o = r[-1]
            conf.append(OrderedDict(
                zip(("id", "?", "rate","trig.mode","trig.src","trig.delay","?","max.volt","trig.thresh"), 
                    r[:-1])))
        return [("ver", ver),
                ("ad_type", {2:"16-bit signed"}[ad_type] ),
                ("num", num),
                ("size", size),
                ("conf", conf)]

    _dummy = lambda data: [("data", data)]

    _dummy_ver = lambda data: [ ("ver", unpack("H",data)[0]), ("data", data[2:])]
    
    _messages = {
        1: ("AE hit/Event Data", _dummy),
        2: ("Time Demand Data", _dummy),
        7: ("User Comments/Test Label", _dummy),
        9: ("Not Used", _dummy),
        11: ("Reset Real Time Clock", _dummy),
        15: ("Abort acquisition/transfer", _dummy),
        38: ("Test Info", _dummy),
        41: ("ASCII Product Definition", 
             lambda data: [ ("ver", unpack("xH",data)[1]), ("text", data[3:].rstrip("\r\n\x00\x1a"))]
            ),
        42: ("Hardware Setup", _dummy),
        44: ("Location Setup", _dummy),
        49: ("Product Specific Setup & Configuration Information", _dummy),
        
        99: ("Time and Date of Test Start", 
             lambda data: [("date", data.rstrip("\n\x00"))]
            ),
        107: ("Reserved", _dummy),
        110: ("Define Group Parametric Channels", _dummy),
        116: ("Not Used", _dummy),
        128: ("Begin Test", _dummy),
        129: ("Stop Test", _dummy),
        130: ("Pause Test", _dummy),
        137: ("Analog Filter Selection", _dummy),
        #173: ("AEDSP waveform recording", dummy),
        (173, 1): ("Digital AE Waveform Data", _msg_173_1),

        (174, 1): ("Digital AE Waveform Data", _msg_174_1),
        (174, 20): ("Pre-Amp Gain", _dummy_ver),
        (174, 23): ("Set Gain", _dummy_ver),
        (174, 42): ("Hardware Setup", _msg_174_42),
        (174, 106): ("Begin a Group Setup", _msg_174_106),
        (174, 174): ("??", _msg_174_174),
    }


from .ae import Event, Data

class WFS(Data):
    def __init__(self, fname):
        self.fname = fname
        fh = open(fname,"rb")
        import mmap, sys, time
        start = time.time()
        self.mmap = mmap = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        
        offset = 0
        print_offset = 0
        total_size = len(mmap)
        reading_data = 'not yet'
        data_offset = None
        num_data = 0

        self.meta = OrderedDict()
        from struct import unpack_from
        while offset < total_size:
            length, id1, id2 = unpack_from("HBB", mmap, offset)
           
            if id1 == 174 and id2 == 1:
                assert length == 2048+28
                assert reading_data != 'done'
                num_data += 1
                if reading_data == 'not yet':
                    data_offset = offset
                    reading_data = 'yes'
            else:
                if reading_data == 'yes':
                    reading_data = 'done'
                
                if id1 in (173, 174):
                    msg_id, msg_offset, msg_length = (id1,id2), offset+4, length-2
                else:
                    msg_id, msg_offset, msg_length = id1, offset+3, length-1

                data = mmap[msg_offset:msg_offset+msg_length]
                msg = Msg(msg_id, data)
                self.meta[msg.label] = Msg(msg_id, data)

            offset += length + 2

            if offset >= print_offset:
                print_offset += 2**20
                sys.stderr.write("\r{:.0f}%".format(offset*100./total_size))
                sys.stderr.flush()
        sys.stderr.write("\r{:.0f}% {:.2f}s\n".format( 100., time.time()-start) )

        self.timescale = 1e-3/self.meta['Hardware Setup']['conf'][0]['rate']
        self.datascale = self.meta['Hardware Setup']['conf'][0]['max.volt']/32768.
        
        from numpy import frombuffer
        a = frombuffer(mmap, dtype="h", offset=data_offset, count=num_data*(2048+28+2)/2)
        a.shape = (num_data, (2048+28+2)/2)
        self.data = a[:,-1024:]

    def __str__(self):
        return "\n".join( str(m) for m in self.meta.values())

    def plot(self, **kwargs):
        from .plot import Downsampler
        return Downsampler(self.data, self.timescale, self.datascale, **kwargs)

    def get_events(self, thresh, hdt=1000, dead=1000):
        from ._wfs_events import _get_events

        import time, sys
        _thresh = int(thresh/self.datascale)
        _hdt = int(hdt) #TODO convert from microseconds
        _dead = int(dead)
        data = self.data
        assert len(data.shape) == 2
        ret = []
        def callback(a,b):
            ret.append( Event(a, b+_hdt, data.flat[a:b+_hdt]*self.datascale, thresh) )
        
        start = time.time()
        _get_events(callback, data, _thresh, _hdt, _dead)
        sys.stderr.write("\r{:.0f}% {:.2f}s\n".format(100., time.time()-start))
        sys.stderr.write("{} events\n".format(len(ret)))
        return ret

def build_ext(location="."):
    from scipy.weave.ext_tools import ext_module, ext_function
    
    mod = ext_module("_wfs_events")

    # define parameter types
    threshold = hdt = dead = 0
    from numpy import empty
    data = empty(shape=(0,0), dtype='h')
    def callback(a,b):
        pass

    func = ext_function('_get_events', r'''
    
    int event_start = -1;
    int last = -1;
    for (int i=0; i<Ndata[0]*Ndata[1]; ++i){
        short d = DATA2(i/Ndata[1],i%Ndata[1]); 
        if (d > threshold) {
            if (last == -1) {
                last = event_start = i;

            } else if (i > last+hdt+dead) {
                PyObject_CallFunction(py_callback, "(ii)", event_start, last); 

                last = event_start = i;

            } else if (i < last+hdt) {
                last = i;
            }
        }
        if (i%1000000 == 0){
            fprintf(stderr, "\r%.0f%%", i*100./Ndata[0]/Ndata[1]);
        }
    }

    if (last != -1){
        PyObject_CallFunction(py_callback, "(ii)", event_start, last); 
    }

    ''', ['callback', 'data', 'threshold', 'hdt', 'dead'])
    mod.add_function(func)
    
    return mod.setup_extension(location=location)


if __name__ == "__main__":
    import sys
    f = WFS(sys.argv[1])
    print f
    print "data.shape:", f.data.shape
    print "rate:", f.meta['Hardware Setup']['conf'][0]['rate']
    print "max.volt:", f.meta['Hardware Setup']['conf'][0]['max.volt']
    
    f.get_events(thresh=0.02, hdt=1000, dead=1000)
