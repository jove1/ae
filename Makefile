
PYTHON?=python

all: sdist wininst

sdist: version
	$(PYTHON) setup.py build sdist

wininst: wininst32 wininst64
	ls -1 dist/*.exe

wininst32:
	$(PYTHON) setup.py build --plat-name=win32 --cross-ver=2.7 --cross-dir=cross/python.win32-2.7 bdist_wininst --wininst-exe-path=cross/wininst

wininst64:
	$(PYTHON) setup.py build --plat-name=win-amd64 --cross-ver=2.7 --cross-dir=cross/python.win-amd64-2.7 bdist_wininst --wininst-exe-path=cross/wininst

develop:
	#$(PYTHON) setup.py develop
	$(PYTHON) -c "import setuptools; execfile('setup.py')" develop

test:
	#$(PYTHON) setup.py test
	$(PYTHON) -c "import setuptools; execfile('setup.py')" test

clean:
	rm -Rf build dist/ae-* ae.egg-info


.PHONY: test clean
