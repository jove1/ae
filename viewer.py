#!/usr/bin/python

if __name__ == "__main__":
    
    import sys, ae
    from matplotlib.pyplot import *
    figure(figsize=(8,4))
    subplots_adjust(0.12,0.15,0.98,0.98)
    [ ae.open(fname).plot(label=fname) for  fname in sys.argv[1:]]
    ae.xpan()
    legend()
    xlabel("time [s]")
    ylabel("amplitude [V]")
    grid()
    show()
