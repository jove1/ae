#!/usr/bin/python

if __name__ == "__main__":
    import sys, ae
    from matplotlib.pyplot import *
    for fname in sys.argv[1:]:
        x = ae.open(fname)
        print x
        figure(fname, figsize=(8,4))
        subplots_adjust(0.12,0.15,0.98,0.95)

        for ch in range(x.channels)[::-1]:
            x.plot(channel=ch, label="ch#{}".format(ch))
        ae.xpan()
        legend()
        grid()
        xlabel("time [{}]".format(x.timeunit))
        ylabel("amplitude [{}]".format(x.dataunit))
    show()
