#!/usr/bin/python

if __name__ == "__main__":
    import sys, ae
    x = ae.open(sys.argv[1])
   
    events = x.get_events(0.02)
    print events.size, "events"

    try:
        is_interactive = get_ipython().config['TerminalIPythonApp']['force_interact']
    except (NameError, KeyError):
        import sys
        is_interactive = sys.flags.interactive

    from pylab import *
    if is_interactive:
        ion()

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
    mask = events.rise_times<events.pre 
    mask[:-1] |= mask[1:]
    events[mask].plot()
    grid()
    ae.xpan()

    #for ax in events[argsort(events.maxima)[-10:]].plot( ax=(figure().gca() for x in xrange(10)) ):
    #    ax.grid()

    figure()
    for ax in events[argsort(events.maxima)[-10:]].plot( ax=(subplot(3,3,x+1) for x in xrange(9)) ):
        ax.grid()
        ax.set_xticks([])
        ax.set_yticks([])
    tight_layout()

    show()
