

SO=build/lib.linux-i686-2.7/ae/event_detector.so
PYD=build/lib.win32-2.7/ae/event_detector.pyd


all: so pyd

so: $(SO)
pyd: $(PYD)

$(SO):ae/event_detector.c
	python setup.py build

$(PYD):ae/event_detector.c
	wine "C:\Python27\python" setup.py build -c mingw32

clean:
	rm -Rf build/
