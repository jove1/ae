#!/usr/bin/python

def scroll_callback(event):
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
    
if __name__ == "__main__":
    
    from matplotlib.pyplot import *
    figure(figsize=(8,4))
    subplots_adjust(0.12,0.15,0.98,0.98)
    
    import sys, ae
    plots = [ ae.open(fname).plot(label=fname) for  fname in sys.argv[1:]]
    
    gca().figure.canvas.mpl_connect('scroll_event', scroll_callback)
    #TODO x-dragging
    legend()
    xlabel("time [s]")
    ylabel("amplitude [V]")
    grid()
    show()
