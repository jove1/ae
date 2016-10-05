#!/usr/bin/env python
#encoding: utf8

import pyae # fail early if not found

import matplotlib
matplotlib.use('TkAgg')

import logging
logging.basicConfig(level=logging.INFO)

from traits.trait_base import ETSConfig
ETSConfig.toolkit = 'null'

from traits.api import provides, HasTraits, List, cached_property, Enum, Any, Property

import pyae.cluster.ask
class ASKC(pyae.cluster.ask.ASKC):
    filetype = Enum('h5', ['h5', 'wfs', 'custom'])

from pyae.interfaces.i_read import IRead
import numpy as np

@provides(IRead)
class AEReader(HasTraits):
    d = Any()
    meta = Property(List(), depends_on=['d'])

    def c_read(self, num_ticks, win_size, channel=1, pad=False):
        assert num_ticks == 0
        assert pad == False
        buf = np.array([])
        source = self.d.iter_blocks(channel=channel-1)
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

def gui(top=None, reader=None):
    top = top or tk.Tk()
    top.title("ASKC Run")

    data = [
        ('param_txt_filename',      "Output file:",         tk.StringVar(value="askc_results.csv")),
        ('fs',                      "Sample rate [Mhz]:",   tk.DoubleVar(value=1.0)),
        ('channel',                 "Channel:",             tk.IntVar(value=1)),
        ('frame_size',              "Frame size:",          tk.IntVar(value=1024)),
        ('n_signals',               "Num. signals:",        tk.IntVar(value=5000)),
        ('noise_frames',            "Noise frames:",        tk.IntVar(value=0)),
        ('how_many_noise_sigma',    "Noise sigma:",         tk.IntVar(value=2)),
        ('significance_threshold',  "Sign. thresh.:",       tk.IntVar(value=150)),
        ('max_num_clusters',        "Max. clusters:",       tk.IntVar(value=10)),
        ('distance',                "Distance:",            tk.StringVar(value='KL')),

        ('nperseg',                 "nperseg:",             tk.IntVar(value=256)),
        ('nfft',                    "nfft:",                tk.IntVar(value=512)),
        ('noverlap',                "noverlap:",            tk.IntVar(value=64)),
        
        ('do_filter',               "Filter:",              tk.StringVar(value='bandpass')),
        ('cutoff_freq',             "Cutoff [kHz]:",        tk.DoubleVar(value=60.0)),
        ('low_cutoff_freq',         "Low cutoff [kHz]:",    tk.DoubleVar(value=300.0)),
        ('high_cutoff_freq',        "High cutoff [kHz]:",   tk.DoubleVar(value=500.0)),
    ]

    if reader is None:
        data.insert(0, 
        ('fname',                   "Input file:",          tk.StringVar())
        )

    vals={k:v for k,_,v in data}
    lbls={k:v for k,v,_ in data}

    def openfile(event=None):
        fname = tkFileDialog.askopenfilename(parent=top, filetypes=[('WFS', '.wfs')])
        vals['fname'].set(fname)

    def savefile(event=None):
        fname = tkFileDialog.asksaveasfilename(parent=top, initialfile=vals['param_txt_filename'].get())
        vals['param_txt_filename'].set(fname)

    row = 0
    frame = tk.LabelFrame(top, text="Basic parameters")
    frame.pack(padx=5, pady=5, fill=tk.X)
    
    if reader is None:
        for k in ['fname']:
            tk.Label(frame, text=lbls[k]).grid(row=row, sticky="W")
            tk.Entry(frame, text=vals[k], state="readonly").grid(row=row, column=1)
            tk.Button(frame, text="...", command=openfile).grid(row=row, column=2)
            row += 1

    for k in ['param_txt_filename']:
        tk.Label(frame, text=lbls[k]).grid(row=row, sticky="W")
        tk.Entry(frame, text=vals[k], state="readonly").grid(row=row, column=1)
        tk.Button(frame, text="...", command=savefile).grid(row=row, column=2)
        row += 1

    for k in ['fs', 'channel', 'frame_size', 'n_signals', 'noise_frames', 'how_many_noise_sigma', 'significance_threshold', 'max_num_clusters']:
        tk.Label(frame, text=lbls[k]).grid(row=row, sticky="W")
        tk.Entry(frame, text=vals[k]).grid(row=row, column=1, sticky="W")
        row += 1

    for k in ['distance']:
        tk.Label(frame, text=lbls[k]).grid(row=row, sticky="W")
        tk.OptionMenu(frame, vals[k], "KL", "Euclidean", "SAM").grid(row=row, column=1, sticky="W")
        row += 1

    row = 0
    frame = tk.LabelFrame(top, text="FFT")
    frame.pack(padx=5, pady=5, fill=tk.X)

    for k in ['nperseg', 'nfft', 'noverlap']:
        tk.Label(frame, text=lbls[k]).grid(row=row, sticky="W")
        tk.Entry(frame, text=vals[k]).grid(row=row, column=1, sticky="W")
        row += 1

    row = 0
    frame = tk.LabelFrame(top, text="Filter")
    frame.pack(padx=5, pady=5, fill=tk.X)
    
    tk.Radiobutton(frame, text="None", variable=vals['do_filter'], value="none").grid(row=row, sticky="W", columnspan=3)
    row += 1
    
    tk.Radiobutton(frame, text="Highpass", variable=vals['do_filter'], value="highpass").grid(row=row, sticky="W", columnspan=3)
    row += 1
    for k in ['cutoff_freq']:
        tk.Label(frame, text=lbls[k]).grid(row=row, column=1, sticky="W")
        tk.Entry(frame, text=vals[k]).grid(row=row, column=2, sticky="W")
        row += 1

    tk.Radiobutton(frame, text="Bandpass", variable=vals['do_filter'], value="bandpass").grid(row=row, sticky="W", columnspan=3)
    row += 1
    for k in ['low_cutoff_freq', 'high_cutoff_freq']:
        tk.Label(frame, text=lbls[k]).grid(row=row, column=1, sticky="W")
        tk.Entry(frame, text=vals[k]).grid(row=row, column=2, sticky="W")
        row += 1

    frame.grid_columnconfigure(0,minsize=25)
    
    def ok(event=None):
        import traceback
        try:
            kwargs = {k:v.get() for k,v in vals.iteritems()}
            if reader is None:
                 open(kwargs['fname']).close()
                 kwargs['filetype'] = 'wfs'
            else:
                kwargs['_reader'] = reader
                kwargs['filetype'] = 'custom'
 
            open(kwargs['param_txt_filename'], "w").close()
            askc_run(**kwargs)
        except IOError,e:
            tkMessageBox.showwarning("Bad input", "[Errno {0.errno}] {0.strerror}: {0.filename!r}".format(e), parent=top)
        except ValueError,e:
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
