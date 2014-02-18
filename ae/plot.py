#!/usr/bin/python

import numpy as np

class Downsampler:
    def __init__(self, data, xscale=1, yscale=1, ax=None, num=2048, **kwargs):
        if len(data.shape) > 1: # discontigous data stored as 2d array
            import sys, time
            start =  time.time()
            if 0:
                self.mins = data.min(axis=-1)
                self.maxs = data.max(axis=-1)
            else:
                # TODO try to do this way while reading the file
                self.mins = np.empty(data.shape[0])
                self.maxs = np.empty(data.shape[0])  
                step = max(data.shape[0]//50, 1)
                for i in xrange(0, data.shape[0]+step, step):
                    data[i:i+step].min(axis=-1, out=self.mins[i:i+step])
                    data[i:i+step].max(axis=-1, out=self.maxs[i:i+step])
                    sys.stderr.write("\r{:.0f}%".format(i*100./data.shape[0]))
                    sys.stderr.flush()
            sys.stderr.write("\r{:.0f}% {:.2f}s\n".format(100., time.time()-start))
            self.data = data
        else:
            self.data = data

        self.xscale = xscale
        self.yscale = yscale
        self.num = num

        if ax is None:
            from matplotlib.pyplot import gca
            ax = gca()
        self.line, = ax.plot([],[], **kwargs)

        if len(data.shape) > 1:
            ax.set_ylim(self.mins.min()*yscale, self.maxs.max()*yscale)
        else:
            ax.set_ylim(self.data.min()*yscale, self.data.max()*yscale)
        
        ax.callbacks.connect('xlim_changed', self._update)
        ax.set_xlim(0,(self.data.size-1)*xscale)

    def _update(self, ax):
        self.line.set_data(*self.resample(*ax.viewLim.intervalx))
        ax.figure.canvas.draw_idle()

    def resample(self, a, b):
        a = int(np.floor(a/self.xscale))
        b = int(np.ceil(b/self.xscale))
        s = max((b+1-a)//self.num, 1)
        #print "resample", s, b+1-a

        a = max(a, 0)
        b = min(b, self.data.size)
        
        if len(self.data.shape) > 1:
            r = self.data.shape[-1]
            if s > r:
                ss = s//r
                aa = a//r
                bb = b//r
                s = s//r*r
                a = a//r*r
                b = b//r*r
                mins = self.mins[aa//ss*ss:bb//ss*ss]
                mins.shape = (bb//ss-aa//ss, ss)
                maxs = self.maxs[aa//ss*ss:bb//ss*ss]
                maxs.shape = (bb//ss-aa//ss, ss)
            else:
                d = self.data.flat[a//s*s:b//s*s]
                d.shape = (b//s-a//s, s)
                mins = maxs = d

        else:
            d = self.data[a//s*s:b//s*s]
            d.shape = (b//s-a//s, s)
            mins = maxs = d

        x = np.empty(2*mins.shape[0])
        y = np.empty(2*mins.shape[0])
        x[::2] = x[1::2] = np.arange(a//s*s,b//s*s,s)*self.xscale
        mins.min(axis=-1, out=y[::2])
        maxs.max(axis=-1, out=y[1::2])
        y *= self.yscale
        return x,y


