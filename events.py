#!/usr/bin/python

if __name__ == "__main__":
    import sys, ae
    x = ae.open(sys.argv[1])
   
    events = x.get_events(0.02)
    print len(events), "events"



    from pylab import *
    
    figure(figsize=(12,6))
    for i,l in enumerate("durations energies maxima rise_times counts".split()):
        subplot(2,3,i+1)
        title(l)
        hist, bins = ae.loghist(getattr(events,l))
        plot( (bins[1:]+bins[:-1])/2, hist, "o")
        loglog()
        grid(True)
    tight_layout()
    
    figure()
    for e in events:
        #if e.data.size > 10000:
        if 1:
            plot(e.start + arange(-events.pre, e.data.size-events.pre), e.data)
            axhline(events.thresh,color="k")
            axvspan(e.start, e.start+e.data.size-events.post-events.pre-1, color="k", alpha=0.2)
            print e.count(events.thresh)
            break
    grid()

    show()
