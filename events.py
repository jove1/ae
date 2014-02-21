#!/usr/bin/python

if __name__ == "__main__":
    import sys, ae
    x = ae.open(sys.argv[1])
   
    events = x.get_events(0.02)
    print events.size, "events"

    from pylab import *
    
    figure(figsize=(12,6))
    for i,l in enumerate("durations energies maxima rise_times counts".split()):
        subplot(2,3,i+1)
        title(l.replace("_"," "))
        hist, bins = ae.loghist(getattr(events,l))
        plot( (bins[1:]+bins[:-1])/2, hist, "o")
        loglog()
        grid(True)
    tight_layout()
    
    figure()
    #for e in events[events.maxima>1]:
    mask = events.rise_times<events.pre 
    mask[:-1] |= mask[1:]
    events[mask].plot()
    grid()
    ae.xpan()

    show()
