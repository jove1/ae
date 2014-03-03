#!/usr/bin/env python
#encoding: utf8

import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import Tkinter as Tk
import ttk
import tkMessageBox
import tkFileDialog
import tkSimpleDialog

import ae

class Dialog(tkSimpleDialog.Dialog):
    def __init__(self, master, title, fields):
        self.fields = fields
        tkSimpleDialog.Dialog.__init__(self, master, title)
    
    def body(self, master):
        self.entries = []
        for i,(label,value) in enumerate(self.fields):
            l = Tk.Label(master, text=label)
            l.grid(row=i, column=0)

            e = Tk.Entry(master)
            e.grid(row=i, column=1)
            e.delete(0, Tk.END)
            e.insert(0, str(value))
            self.entries.append(e)
        return self.entries[0]

    def validate(self):
        values = []
        for e,(_, value) in zip(self.entries, self.fields):
            try:
                v = type(value)(e.get())
            except ValueError,e:
                tkMessageBox.showwarning("Bad input", e.message)
                return 0
            else:
                values.append(v)
        self.result = values
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
        if fname:
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
        self.toolbar = NavigationToolbar2TkAgg(canvas, frame)

        canvas.mpl_connect('key_press_event', self.on_key_press)
        canvas.mpl_connect('scroll_event', self.on_scroll)
        
        self.progressbar = ttk.Progressbar(self.root, orient='horizontal', mode='determinate', )
        self.progressbar.pack(side=Tk.BOTTOM, fill=Tk.X)
        
        self.data = None
        self.root.update() # needed on windows for events to work

    def on_scroll(self, event):
        ax = event.inaxes
        if ax is None:
            return
    
        if event.key == 'shift':
            dir = {'up': -0.2, 'down': 0.2}[event.button]
            a,b = ax.viewLim.intervalx
            ax.set_xlim([a+dir*(b-a),
                         b+dir*(b-a)])
        else:
            scale = {'up': 0.8, 'down': 1.25}[event.button]
            x = event.xdata
            a,b = ax.viewLim.intervalx
            ax.set_xlim([x+scale*(a-x),
                         x+scale*(b-x)])
        ax.figure.canvas.draw()

    def on_key_press(self, event):
        if event.key == 'a' or event.key == "alt+a" and event.inaxes:
            ax = event.inaxes
            ax.relim()
            l = 1.1*max(abs(ax.dataLim.y0), abs(ax.dataLim.y1))
            ax.set_ylim(-l,l)
            ax.figure.canvas.draw()
        else:
            key_press_handler(event, event.canvas, self.toolbar)

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
        
        _, samples = self.data.iter_blocks(stop=1000).next()
        thresh = samples.std()*self.data.datascale*5
        from math import log10
        thresh = round(thresh,int(-log10(thresh)+2))
        
        d = Dialog(self.root, "Get Events", [
            ("Threshold [{}]:".format(self.data.dataunit), thresh),
            ("HDT [{}]:".format(self.data.timeunit), 0.001),
            ("Dead time [{}]:".format(self.data.timeunit), 0.001),
            ("Pre-trigger [{}]:".format(self.data.timeunit), 0.001),
            ("Channel:", 0)
            ])

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
