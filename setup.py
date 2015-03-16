#!/usr/bin/env python

from distutils.core import setup, Extension
#from setuptools import setup, Extension
import sys

extra_scripts = []
if "bdist_wininst" in sys.argv:
    extra_scripts.append("src/ae_post_install.py")
    extra_scripts.append("src/AEViewer.ico")

cmdclass = {}
if "cross" in sys.argv:
    sys.argv.remove("cross")
    from cross import cross_cmdclass as cmdclass

execfile('src/ae/version.py') # __version__

setup(
    name = 'ae',
    version = __version__,
    description = 'Python accoustic emission tools',
    author = 'Jozef Vesely',
    author_email = 'vesely@gjh.sk',
    url = 'http://github.com/jove1/ae',

    packages = ['ae'],
    package_dir = {'': 'src'},
    scripts = ['src/AEViewer.py'] + extra_scripts,
    ext_modules = [
        Extension('ae.event_detector',
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            sources=['src/ae/event_detector.c'],
        ),
    ],
    requires = ['numpy'],
    options = {
        "bdist_wininst": {
            "install_script": "ae_post_install.py",
            "user_access_control": "auto",
        }
    },
    cmdclass = cmdclass,
)
