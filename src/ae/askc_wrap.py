#!/usr/bin/env python
#encoding: utf8

import pyae # fail early if not found

import matplotlib
matplotlib.use('TkAgg')

import logging
logging.basicConfig(level=logging.INFO)

from traits.trait_base import ETSConfig
ETSConfig.toolkit = 'null'

from traits.api import provides, HasTraits, List, cached_property, Enum, Any, Property, Int

import pyae.cluster.ask
class ASKC(pyae.cluster.ask.ASKC):
    filetype = Enum('h5', ['h5', 'wfs', 'custom'])

from pyae.interfaces.i_read import IRead
import numpy as np

@provides(IRead)
class AEReader(HasTraits):
    d = Any()
    skip_frames = Int()
    meta = Property(List(), depends_on=['d'])

    def c_read(self, num_ticks, win_size, channel=1, pad=False):
        assert num_ticks == 0
        assert pad == False
        buf = np.array([])
        source = self.d.iter_blocks(start=win_size*self.skip_frames, channel=channel-1)
        pos, data = source.next()
        offset = win_size*self.skip_frames-pos
        print "seek", pos, win_size*self.skip_frames, offset
        buf = np.append(buf, data.flat[offset:])
        try:
            while True:
                while len(buf) < win_size:
                    _, data = source.next()
                    buf = np.append(buf, data)
                while len(buf) >= win_size:
                    yield buf[:win_size]
                    buf = buf[win_size:]
        except StopIteration:
            yield buf
    
    @cached_property
    def _get_meta(self):
        return [ self.d.channels,
                 {i+1: self.d.size for i in xrange(self.d.channels)},
                 {i+1: 1./self.d.timescale for i in xrange(self.d.channels)} ]

def askc_run(**kwargs):
    from pyae.signal_process.norms.NormsWrap import NormsWrap
    def simple_norm(values):
        s = np.sum(values)
        if s == 0:
            return values
        return values/s
    norm = NormsWrap(func=simple_norm)
    norm.set_param()

    kwargs['cutoff_freq'] *= 1e3
    kwargs['low_cutoff_freq'] *= 1e3
    kwargs['high_cutoff_freq'] *= 1e3
    kwargs['fs'] *= 1e6
    if kwargs['do_filter'] != 'none':
        kwargs['filter_type'] = kwargs['do_filter']
        kwargs['do_filter'] = True
    else:
        kwargs['do_filter'] = False

    import pyae.spatial.distance # KL, Euclidean or SAM
    kwargs['distance'] = getattr(pyae.spatial.distance, 
                                 kwargs['distance'])()

    from pyae.signal_process.utils import FFTParams
    kwargs['fft_params'] = FFTParams(
        nperseg=kwargs.pop('nperseg'), 
        nfft=kwargs.pop('nfft'), 
        noverlap=kwargs.pop('noverlap')
    )

    askc = ASKC(
        norm = norm,

        # Parameter extraction
        extract_params = True,
        save_clustering_result = True,

        # If try or not to minimize the effect of outliers
        handle_outliers = True,

        # Use the noise properties to enhance the rest of the 
        # time series. It is suggested just when the noise is 
        # suppose to be stationary during the whole time lasted
        # by the test.
        use_spectral_subtraction = True,
        
        domain = 'freq',
        _ask_version_ = 2,

        **kwargs
    )
    askc.fit()
    print askc.n_signals
    print askc.R_noise

    ask = askc.ask
    f = askc.freq
    t = askc.t
    df = askc._df

    for k in ask.clusters:
        print 'cluster "%s": %s' % (k, ask.clusters[k].nEle)


    import matplotlib.pyplot as plt
    from pyae.plots.utils import FixedOrderFormatter

    import seaborn as sns
    sns.heatmap(askc.ask.clusters_pairwise_dist);

    # In[18]:

    fig = plt.figure(figsize=(8,6))
    #y_max = np.max(ask.clusters_mat)
    #y_min = np.min(ask.clusters_mat)
    ax = fig.add_subplot(111)
    for k in ask.clusters:
        if askc.domain == 'time':
            f = np.arange(ask.clusters[k].centroid.shape[0])
        ax.plot(f,ask.clusters[k].centroid,label='Cluster "%s" (%s)' % (k,ask.clusters[k].nEle));
        #ax.set_ylim((y_min,y_max))
        #ax.xaxis.set_major_formatter(FixedOrderFormatter(3))
        if askc.domain == 'time':
            ax.set_xlabel('$samples$');
        else:
            ax.set_xlabel('$f (KHz)$');
        ax.set_ylabel('$Amp.$');
        #for item in ([plt.gca().title, plt.gca().xaxis.label, plt.gca().yaxis.label]):
        #    item.set_fontsize(16)
        #    item.set_fontweight('bold')
            
        #for label in plt.gca().get_xticklabels():
        #    label.set_fontsize(16)
            
        #for label in plt.gca().get_yticklabels():
        #    label.set_fontsize(16)
        ax.legend(loc='best')

    plt.tight_layout()


    # In[19]:

    fig = plt.figure(figsize=(8,3*ask.n_clusters))
    #y_max = np.max(ask.clusters_mat)
    #y_min = np.min(ask.clusters_mat)
    for k in ask.clusters:
        ax = fig.add_subplot(ask.n_clusters,1,k+1)
        if askc.domain == 'time':
            f = np.arange(ask.clusters[k].centroid.shape[0])
        ax.plot(f,ask.clusters[k].centroid,label='Cluster "%s" (%s)' % (k,ask.clusters[k].nEle));
        #ax.set_ylim((y_min,y_max))
        #ax.xaxis.set_major_formatter(FixedOrderFormatter(3))
        if askc.domain == 'time':
            ax.set_xlabel('$samples$');
        else:
            ax.set_xlabel('$f (KHz)$');
        ax.set_ylabel('$Amp.$');
        #for item in ([plt.gca().title, plt.gca().xaxis.label, plt.gca().yaxis.label]):
        #    item.set_fontsize(16)
        #    item.set_fontweight('bold')
            
        #for label in plt.gca().get_xticklabels():
        #    label.set_fontsize(16)
            
        #for label in plt.gca().get_yticklabels():
        #    label.set_fontsize(16)
        ax.legend(loc='best')

    plt.tight_layout()


    # In[20]:

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    for k in ask.clusters:
        cwa = np.cumsum(np.array(ask.clusters_waveindex_ass) == k)
        ax.plot(t,cwa,label="cluster %s (%s)" % (k,ask.clusters[k].nEle));
    ax.legend(loc='best');


    # In[21]:

    fig = plt.figure(figsize=(8,3*ask.n_clusters))
    for k in ask.clusters:
        cwa = np.cumsum(np.array(ask.clusters_waveindex_ass) == k)
        ax = fig.add_subplot(len(ask.clusters.keys()),1,k+1)
        ax.plot(t,cwa,label="cluster %s (%s)" % (k,ask.clusters[k].nEle));
        ax.legend(loc='best')


    # In[22]:

    # Available parameters: 
    #
    #                'Time',
    #                'Cluster',
    #                'Energy',
    #                'Mean',
    #                'Var',
    #                'MedianFreq',
    #                'Kur',
    #                'Paek',
    #                'PeakLoc',
    #                'Entropy',
    #                'EnergyT',
    #                'MaxAmpT',
    #                'MeanT',
    #                'VarT',
    #                'SkewT',
    #                'KurT'

    x = 'MedianFreq'
    y = 'Kur'


    # In[23]:

    from itertools import cycle
    colors = cycle('rcmbgk')
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    for (cl,color) in zip(xrange(ask.n_clusters),colors):
        if np.sum(df['Cluster']==cl) > 5:
            ax.scatter(df[x][df['Cluster']==cl],
                       df[y][df['Cluster']==cl],
                       c=color, marker='o',s=30, label='cl %s' % cl);
            fig_cl = plt.figure(figsize=(6,4))
            axc = fig_cl.add_subplot(111)
            axc.scatter(df[x][df['Cluster']==cl],
                       df[y][df['Cluster']==cl],
                       c=color, marker='o',s=30, label='cl %s (%s)' % (cl,np.sum(df['Cluster']==cl)));
            axc.set_title('%s vs %s' % (x,y));
            axc.set_xlabel('$F_m (Hz)$');
            axc.set_ylabel('$K$');
            #axc.set_ylim((-3,30))
            #axc.set_xlim((25e4,60e4))
            axc.legend(loc='best');
    ax.set_title('%s vs %s' % (x,y));
    ax.set_xlabel('$F_m (Hz)$');
    ax.set_ylabel('$K$');
    #ax.set_ylim((-3,10))
    ax.legend(loc='best');


    # In[24]:

    x = 'MedianFreq'
    y = 'MaxAmpT'


    # In[25]:

    from itertools import cycle
    colors = cycle('rcmbgk')
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    for (cl,color) in zip(xrange(ask.n_clusters),colors):
        if np.sum(df['Cluster']==cl) > 5:
            ax.scatter(df[x][df['Cluster']==cl],
                       df[y][df['Cluster']==cl],
                       c=color, marker='o',s=30, label='cl %s' % cl);
            fig_cl = plt.figure(figsize=(6,4))
            axc = fig_cl.add_subplot(111)
            axc.scatter(df[x][df['Cluster']==cl],
                       df[y][df['Cluster']==cl],
                       c=color, marker='o',s=30, label='cl %s' % cl);
            axc.set_title('%s vs %s' % (x,y));
            axc.set_xlabel('$F_m (Hz)$');
            axc.set_ylabel('$Max Amp.$');
            #axc.set_ylim((-500,5000))
            #axc.set_xlim((25e4,60e4))
            axc.legend(loc='best');
    ax.set_title('%s vs %s' % (x,y));
    ax.set_xlabel('$F_m (Hz)$');
    ax.set_ylabel('$Max Amp.$');
    #ax.set_ylim((-500,5000))
    ax.legend(loc='best');

    plt.show(False)

    return askc

import Tkinter as tk
import tkFileDialog
import tkMessageBox

class Item:
    def __init__(self, item, **kwargs):
        self.item = item
        self.kwargs = kwargs

    def grid(self, *args, **kwargs):
        kwargs.update(self.kwargs)
        self.item.grid(*args, **kwargs)

def make_grid(frame, grid):
    for r,row in enumerate(grid):
        for c,item in enumerate(row):
            if item is None:
                continue
            if isinstance(item, str):
                item = tk.Label(frame, text=item)
            item.grid(row=r, column=c, sticky="W")

class Vals:
    def __init__(self, **kwargs):
        m = {str: tk.StringVar,
             int: tk.IntVar,
             float: tk.DoubleVar}

        for k,v in kwargs.items():
            self.__dict__[k] = m[type(v)](value=v)

    def get(self):
        return {k:v.get() for k,v in self.__dict__.items()}

class Updater:
    def __init__(self, vals, skip, num):
        self.skip_time = tk.DoubleVar(value=skip)
        self.n_time = tk.DoubleVar(value=num)

        self.fs = vals.fs
        self.frame_size = vals.frame_size
        self.skip_frames = vals.skip_signals
        self.n_frames = vals.n_signals
        
        self.updating = False
        
        self.fs.trace("w", self.update_frames)
        self.frame_size.trace("w", self.update_frames)
        self.skip_time.trace("w", self.update_frames)
        self.n_time.trace("w", self.update_frames)

        self.skip_frames.trace("w", self.update_times)
        self.n_frames.trace("w", self.update_times)

        self.update_frames()
    
    def update_frames(self, *args):
        if self.updating:
            return 
        self.updating = True
        try:
            self.skip_frames.set(int(self.skip_time.get()*1e6*self.fs.get()/self.frame_size.get()))
            self.n_frames.set(int(self.n_time.get()*1e6*self.fs.get()/self.frame_size.get()))
        except ValueError:
            pass
        finally:
            self.updating = False

    def update_times(self, *args):
        if self.updating:
            return
        self.updating = True
        try:
            self.skip_time.set(self.skip_frames.get()*self.frame_size.get()/1e6/self.fs.get())
            self.n_time.set(self.n_frames.get()*self.frame_size.get()/1e6/self.fs.get())
        except ValueError:
            pass
        finally:
            self.updating = False


def gui(top=None, reader=None, time_range=(0,5)):
    top = top or tk.Tk()
    top.title("ASKC Run")

    vals = Vals(
        param_txt_filename = "askc_results.csv",
        fs = 1e-6/reader.d.timescale if reader else 1.,
        channel = 1,
        frame_size = 1024,
        skip_signals = 0,
        n_signals = 5000,
        noise_frames = 0,
        how_many_noise_sigma = 2,
        significance_threshold = 150,
        max_num_clusters = 10,
        distance = 'KL',

        nperseg = 256,
        nfft = 512,
        noverlap = 64,

        do_filter = 'bandpass',
        cutoff_freq = 60.,
        low_cutoff_freq = 300.,
        high_cutoff_freq = 500.,
    )
    if not reader:
        vals.fname = tk.StringVar()
   
    u = Updater(vals, time_range[0], time_range[1]-time_range[0])
    
    def openfile(event=None):
        fname = tkFileDialog.askopenfilename(parent=top, filetypes=[('WFS', '.wfs')])
        vals.fname.set(fname)

    def savefile(event=None):
        fname = tkFileDialog.asksaveasfilename(parent=top, initialfile=vals.param_txt_filename.get())
        vals.param_txt_filename.set(fname)
    
    frame = tk.LabelFrame(top, text="Basic parameters")
    frame.pack(padx=5, pady=5, fill=tk.X)
    grid = [
        ("Output file:", tk.Entry(frame, text=vals.param_txt_filename, state="readonly", justify=tk.RIGHT, width=10), tk.Button(frame, text="...", command=savefile) ),
        ("Sample rate:", tk.Entry(frame, text=vals.fs, justify=tk.RIGHT, width=10), "MHz"),
        ("Channel:", tk.Entry(frame, text=vals.channel, justify=tk.RIGHT, width=10)),
        ("Frame size:", tk.Entry(frame, text=vals.frame_size, justify=tk.RIGHT, width=10)),
        ("Skip signals:", tk.Entry(frame, text=vals.skip_signals, justify=tk.RIGHT, width=10), "frames", tk.Entry(frame, text=u.skip_time, justify=tk.RIGHT, width=10), "s"),
        ("Num. signals:", tk.Entry(frame, text=vals.n_signals, justify=tk.RIGHT, width=10), "frames", tk.Entry(frame, text=u.n_time, justify=tk.RIGHT, width=10), "s"),
        ("Noise frames:", tk.Entry(frame, text=vals.noise_frames, justify=tk.RIGHT, width=10)),
        ("Noise sigma:", tk.Entry(frame, text=vals.how_many_noise_sigma, justify=tk.RIGHT, width=10)),
        ("Sign. thresh.:", tk.Entry(frame, text=vals.significance_threshold, justify=tk.RIGHT, width=10)),
        ("Max. clusters:", tk.Entry(frame, text=vals.max_num_clusters, justify=tk.RIGHT, width=10)),
        ("Distance:", tk.OptionMenu(frame, vals.distance, "KL", "Euclidean", "SAM")),
    ]
    if not reader:
        grid.insert(0,
            ("Input file:", tk.Entry(frame, text=vals.fname, state="readonly", justify=tk.RIGHT, width=10), tk.Button(frame, text="...", command=openfile)),
        )
    make_grid(frame, grid)
    
    frame = tk.LabelFrame(top, text="FFT")
    frame.pack(padx=5, pady=5, fill=tk.X)
    grid = [
        ("nperseg:", tk.Entry(frame, text=vals.nperseg, justify=tk.RIGHT, width=10)),
        ("nfft:", tk.Entry(frame, text=vals.nfft, justify=tk.RIGHT, width=10)),
        ("noverlap:", tk.Entry(frame, text=vals.noverlap, justify=tk.RIGHT, width=10)),
    ]
    make_grid(frame, grid)
    
    frame = tk.LabelFrame(top, text="Filter")
    frame.pack(padx=5, pady=5, fill=tk.X)
    grid = [
        (Item( tk.Radiobutton(frame, text="None", variable=vals.do_filter, value="none"), columnspan=4),),
        (Item( tk.Radiobutton(frame, text="Highpass", variable=vals.do_filter, value="highpass"), columnspan=4),),
        (None, "Cutoff:", tk.Entry(frame, text=vals.cutoff_freq, justify=tk.RIGHT, width=10), "kHz"),
        (Item(  tk.Radiobutton(frame, text="Bandpass", variable=vals.do_filter, value="bandpass"), columnspan=4),),
        (None, "Low cutoff:", tk.Entry(frame, text=vals.low_cutoff_freq, justify=tk.RIGHT, width=10), "kHz"),
        (None, "High cutoff:", tk.Entry(frame, text=vals.high_cutoff_freq, justify=tk.RIGHT, width=10), "kHz"),
    ]
    make_grid(frame, grid)
    frame.grid_columnconfigure(0,minsize=25)
    
    def ok(event=None):
        import traceback
        try:
            kwargs = vals.get()

            if not reader:
                 open(kwargs['fname']).close()
                 kwargs['filetype'] = 'wfs'
                 if kwargs.pop('skip_signals') != 0:
                     raise ValueError("Default WFS reader can not skip frames.")
            else:
                kwargs['_reader'] = reader
                reader.skip_frames = kwargs.pop('skip_signals')
                kwargs['filetype'] = 'custom'
 
            open(kwargs['param_txt_filename'], "w").close()
            askc_run(**kwargs)
        except IOError,e:
            traceback.print_exc()
            tkMessageBox.showwarning("Bad input", "[Errno {0.errno}] {0.strerror}: {0.filename!r}".format(e), parent=top)
        except ValueError,e:
            traceback.print_exc()
            tkMessageBox.showwarning("Bad input", e.message, parent=top)
        except Exception,e:
            traceback.print_exc()
            tkMessageBox.showerror("Error", traceback.format_exc(), parent=top)
        else:
            tkMessageBox.showinfo("Finished", "ASKC finished", parent=top)

    b = tk.Button(top, text="Run", width=10, command=ok, default=tk.ACTIVE)
    b.pack(padx=5, pady=5)


if __name__ == "__main__":
    gui()
    tk.mainloop()
