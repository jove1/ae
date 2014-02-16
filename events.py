#!/usr/bin/python

if __name__ == "__main__":
    import sys, ae
    f = ae.open(sys.argv[1])
    events = f.get_events(thresh=0.02, hdt=1000, dead=1000)

    from pylab import *
    
    figure(figsize=(12,6))
    for i,l in enumerate(["max", "max2", "duration", "energy", "rise", "count"]):
        subplot(2,3,i+1)
        title(l)
        hist, bins = ae.loghist([getattr(e,l) for e in events])
        plot( (bins[1:]+bins[:-1])/2, hist, "o")
        loglog()
        grid(True)

    
    figure()
    for e in events:
        if e.duration > 10000:
            plot(e.start+arange(e.data.size), e.data)
    #xlim(0,f.data.size)
    grid()


    show()
