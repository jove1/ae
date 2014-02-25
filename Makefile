
all: so

so:
	python setup.py build

pyd:
	wine "C:\Python27\python" setup.py build -c mingw32

win: 
	wine "C:\Python27\python" setup.py build -c mingw32 bdist_wininst

test: so
	python -m doctest -v ae/event_detector.doctest.rst

clean:
	rm -Rf build/
