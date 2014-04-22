#!/usr/bin/env python
#encoding: utf8

import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import Tkinter as tk
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
            l = tk.Label(master, text=label)
            l.grid(row=i, column=0)

            e = tk.Entry(master)
            e.grid(row=i, column=1)
            e.delete(0, tk.END)
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

class EventsTable(tk.Toplevel):
    columns = [
            ("Start", "starts", 75),
            ("Duration", "durations", 75),
            ("Energy", "energies", 150),
            ("Maximum", "maxima", 150),
            ("Rise time", "rise_times", 75),
            ("Count", "counts", 50),
            ]

    def __init__(self, master, events):
        tk.Toplevel.__init__(self, master)
        self.wm_title("Events")

        self.events = events

        menubar = tk.Menu(self)
        menubar.add_command(label="Save Data", command=self.save)
        self.config(menu=menubar)

        self.tree = ttk.Treeview(self, columns=[name for label,name,width in self.columns])
        ysb = ttk.Scrollbar(self, orient='vertical', command=self.tree.yview)

        self.tree.configure(yscroll=ysb.set)
        self.tree.heading('#0', text='No.')
        self.tree.column('#0', width=50)

        for label, name, width in self.columns:
            self.tree.heading(name, text=label, command=lambda name=name: self.sort(name, True))
            self.tree.column(name, width=width)

        for i,values in enumerate(zip(*(getattr(events, name) for label,name,width in self.columns))):
            iid = self.tree.insert('', 'end', iid=str(i), text=str(i), values=values)

        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=1)

        self.tree.bind("<Double-1>", self.on_doubleclick)
        self.tree.bind("<Button-3>", self.on_rightclick)

        tkMessageBox.showinfo("Events", "Found {} events.".format(events.size), parent=self)

    def save(self):
        from numpy import savetxt, transpose
        fname = tkFileDialog.asksaveasfilename(parent=self, 
                    filetypes=[('Data file', '.txt .dat')])
        if fname:
            savetxt(fname, transpose([getattr(self.events, name) for label, name, width in self.columns]))


    def on_rightclick(self, event):
        row = self.tree.identify_row(event.y)
        if row: # only clicks on headings (use identify_region in Tk 8.6)
            return
        col = self.tree.identify_column(event.x)
        if col == "#0":
            return
        col = int(col.lstrip("#"))
        label, name, _ = self.columns[col-1]
        
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Histogram", command=lambda: Histogram(self, label, getattr(self.events, name)))
        menu.add_command(label="CDF", command=lambda: CDF(self, label, getattr(self.events, name)))
        menu.post(event.x_root, event.y_root)

    def on_doubleclick(self, event):
        item = self.tree.identify_row(event.y)
        if not item:
            return

        item = int(item)

        win = tk.Toplevel(self)
        win.wm_title("Event #{}".format(item))

        fig = Figure(figsize=(4,3), dpi=100)
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, win)
        ax = fig.gca()
        self.events[[item]].plot(ax)
        ax.set_title("Event #{}".format(item))
        ax.grid(True)
        fig.tight_layout()
        win.update()

    def sort(self, name, reverse):
        from numpy import argsort
        order = argsort(getattr(self.events, name))
        if reverse:
            order = order[::-1]
        for i,j in enumerate(order):
            self.tree.move(str(j),'',i)
        self.tree.heading(name, command=lambda: self.sort(name, not reverse))

class Histogram(tk.Toplevel):
    def __init__(self, master, name, data):
        tk.Toplevel.__init__(self, master)
        self.wm_title("Histogram: {}".format(name))

        self.fig = Figure(figsize=(4,3), dpi=100)
        canvas = FigureCanvasTkAgg(self.fig, self)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, self)


        menubar = tk.Menu(self)
        menubar.add_command(label="Save Data", command=self.save)
        self.config(menu=menubar)

        ax = self.fig.gca()
        ax.set_title(name)
        self.hist, self.bins, _ = ae.hist(data, ax=ax, density=True)

        self.update()

    def save(self):
        from numpy import savetxt, transpose
        fname = tkFileDialog.asksaveasfilename(parent=self, 
                    filetypes=[('Data file', '.txt .dat')])
        if fname:
            savetxt(fname, transpose([self.bins[:-1], self.bins[1:], self.hist]))

class CDF(tk.Toplevel):
    def __init__(self, master, name, data):
        tk.Toplevel.__init__(self, master)
        self.wm_title("CDF: {}".format(name))

        self.fig = Figure(figsize=(4,3), dpi=100)
        canvas = FigureCanvasTkAgg(self.fig, self)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, self)


        menubar = tk.Menu(self)
        menubar.add_command(label="Save Data", command=self.save)
        self.config(menu=menubar)

        ax = self.fig.gca()
        ax.set_title(name)
        self.x, self.y = ae.cdf(data)
        
        ax.plot(self.x, self.y, "o")
        ax.loglog()
        ax.grid(True)

        self.update()

    def save(self):
        from numpy import savetxt, transpose
        fname = tkFileDialog.asksaveasfilename(parent=self, 
                    filetypes=[('Data file', '.txt .dat')])
        if fname:
            savetxt(fname, transpose([self.x, self.y]))


class AEViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.wm_title("AEViewer")
        self.make_menu()

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.fig = Figure(figsize=(8,6), dpi=100)
        canvas = FigureCanvasTkAgg(self.fig, frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar2TkAgg(canvas, frame)

        canvas.mpl_connect('key_press_event', self.on_key_press)
        canvas.mpl_connect('scroll_event', self.on_scroll)
        
        self.progressbar = ttk.Progressbar(self.root, orient='horizontal', mode='determinate', )
        self.progressbar.pack(side=tk.BOTTOM, fill=tk.X)
        
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
        menubar = tk.Menu(self.root)

        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.open)
        filemenu.add_command(label="Metadata", command=self.meta)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        menubar.add_command(label="Events", command=self.events)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.about)
        menubar.add_cascade(label="Help", menu=helpmenu)

        self.root.config(menu=menubar)


    def open(self, fname=None):
        
        self.fig.clf()
        self.data = None
        self.data = ae.open(fname, parent=self.root)
        self.data.progress = self.progress
        
        ax = self.fig.gca()
        for ch in range(self.data.channels):
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
        from ae.version import version
        tkMessageBox.showinfo("About AEViewer", "AEViewer {}\nCopyright © 2014 Jozef Vesely".format(version))


    def meta(self):
        if self.data is None:
            return 

        win = tk.Toplevel(self.root)
        win.wm_title("Metadata")

        text = tk.Text(win)
        text.insert(tk.END, str(self.data.meta))
        text.config(state=tk.DISABLED)
        text.pack(fill=tk.BOTH, expand=1)
 
    def events(self):
        if self.data is None:
            return
        
        _, samples = self.data.iter_blocks(stop=1000).next()
        thresh = samples.std()*self.data.datascale[0]*5
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

        EventsTable(self.root, events)

if __name__ == "__main__":
    import sys
    try:
        fname = sys.argv[1]
    except IndexError:
        fname = None

    v = AEViewer()
    v.open(fname)
    
    v.root.mainloop()
