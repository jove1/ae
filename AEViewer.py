#!/usr/bin/env python
#encoding: utf8

import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import Tkinter as Tk
import ttk
import tkMessageBox
import tkFileDialog
import tkSimpleDialog

import ae

class GetEventsDialog(tkSimpleDialog.Dialog):
    
    def body(self, master):
        Tk.Label(master, text="Threshold:").grid(row=0)
        Tk.Label(master, text="HDT:").grid(row=1)
        Tk.Label(master, text="Dead time:").grid(row=2)
        Tk.Label(master, text="Pre-trigger:").grid(row=3)
        Tk.Label(master, text="Channel:").grid(row=4)

        self.thresh = Tk.Entry(master)
        self.hdt = Tk.Entry(master)
        self.dead = Tk.Entry(master)
        self.pre = Tk.Entry(master)
        self.channel = Tk.Entry(master)

        def _set(e, v):
            e.delete(0, Tk.END)
            e.insert(0, v)
        _set(self.hdt, "0.001")
        _set(self.dead, "0.001")
        _set(self.pre, "0.001")
        _set(self.channel, "0")

        self.thresh.grid(row=0, column=1)
        self.hdt.grid(row=1, column=1)
        self.dead.grid(row=2, column=1)
        self.pre.grid(row=3, column=1)
        self.channel.grid(row=4, column=1)

        return self.thresh

    def validate(self):
        try:
            thresh = float(self.thresh.get())
            hdt = float(self.hdt.get())
            dead = float(self.dead.get())
            pre = float(self.pre.get())
            channel = int(self.channel.get())
        except ValueError:
            tkMessageBox.showwarning("Bad input", "Illegal values, please try again")
            return 0
        else:
            self.result = thresh, hdt, dead, pre, channel
            return 1

class Histogram:
    def __init__(self, root, name, data):
        self.win = Tk.Toplevel(root)
        self.win.wm_title("Histogram: {}".format(name))

        self.fig = Figure(figsize=(4,3), dpi=100)
        canvas = FigureCanvasTkAgg(self.fig, self.win)
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, self.win)


        menubar = Tk.Menu(self.win)
        menubar.add_command(label="Save Data", command=self.save)
        self.win.config(menu=menubar)

        ax = self.fig.gca()
        ax.set_title(name.title())
        self.hist, self.bins, _ = ae.hist(data, ax=ax)

        self.win.update()

    def save(self):
        from numpy import savetxt, transpose
        fname = tkFileDialog.asksaveasfilename(parent=self.win, 
                    filetypes=[('Data file', '.txt .dat')])

        savetxt(fname, transpose([self.bins[:-1], self.bins[1:], self.hist]))

class AEViewer:
    def __init__(self):
        self.root = Tk.Tk()
        self.root.wm_title("AEViewer")
        self.make_menu()

        frame = Tk.Frame(self.root)
        frame.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        self.fig = Figure(figsize=(8,6), dpi=100)
        canvas = FigureCanvasTkAgg(self.fig, frame)
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, frame)
        
        self.progressbar = ttk.Progressbar(self.root, orient='horizontal', mode='determinate', )
        self.progressbar.pack(side=Tk.BOTTOM, fill=Tk.X)
        
        self.data = None

    def progress(self, percent, time):
        self.progressbar["value"] = percent
        self.progressbar.update()
    def make_menu(self):
        menubar = Tk.Menu(self.root)

        filemenu = Tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.open)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)


        menubar.add_command(label="Events", command=self.events)

        helpmenu = Tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.about)
        menubar.add_cascade(label="Help", menu=helpmenu)

        self.root.config(menu=menubar)

    def open(self, fname=None):
        if fname is None:
            fname = tkFileDialog.askopenfilename(parent=self.root, 
                    filetypes=[('Any AE file', '.wfs .sdcf'),
                               ('WFS', '.wfs'),
                               ('SDCF','.sdcf'), 
                              ])
        
        self.fig.clf()
        if not fname:
            self.data = None
            return

        self.data = ae.open(fname)
        self.data.progress = self.progress
        
        ax = self.fig.gca()
        for ch in range(self.data.channels)[::-1]:
            self.data.plot(channel=ch, label="ch#{}".format(ch), ax=ax)
        ae.xpan(ax=ax)
        ax.legend()
        ax.grid()
        ax.set_xlabel("time [{}]".format(self.data.timeunit))
        ax.set_ylabel("amplitude [{}]".format(self.data.dataunit))

    def quit(self):
        self.root.quit()     # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent
                             # Fatal Python Error: PyEval_RestoreThread: NULL tstate
    
    def about(self):
        tkMessageBox.showinfo("About AEViewer", "AEViewer © Jozef Vesely 2014")
    
    def events(self):
        if self.data is None:
            return
        d = GetEventsDialog(self.root)
        if d.result is None:
            return
        events = self.data.get_events(*d.result)

        for l in "durations energies maxima rise_times counts".split():
            h = Histogram(self.root, l.replace("_"," "), getattr(events,l) )
        
        tkMessageBox.showinfo("Events", "Found {} events.".format(events.size))

if __name__ == "__main__":
    import sys
    try:
        fname = sys.argv[1]
    except IndexError:
        fname = None

    v = AEViewer()
    v.open(fname)
    
    Tk.mainloop()
