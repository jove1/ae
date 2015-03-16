#!/usr/bin/env python

from distutils.core import setup, Extension
import os.path, sys

extra_scripts = []
install_script = "ae_post_install.py"

if "bdist_wininst" in sys.argv:
    extra_scripts.append(install_script)
    extra_scripts.append("AEViewer.ico")

cmdclass = {}
if "cross" in sys.argv:
    sys.argv.remove("cross")
    from cross import cross_cmdclass as cmdclass

execfile('ae/version.py') # __version__

setup(
    name = 'ae',
    version = __version__,
    description = 'Python accoustic emission tools',
    author = 'Jozef Vesely',
    author_email = 'vesely@gjh.sk',
    url = 'http://github.com/jove1/ae',

    packages = ['ae'],
    scripts = ['AEViewer.py'] + extra_scripts,
    ext_modules = [
        Extension('ae.event_detector',
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            sources=['ae/event_detector.c'],
        ),
    ],
    requires = ['numpy'],
    options = {
        "bdist_wininst": {
            "install_script": install_script,
            "user_access_control": "auto",
        }
    },
    cmdclass = cmdclass,
)
