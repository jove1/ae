#!/usr/bin/env python

from numpy.distutils.core import setup, Extension

extra_scripts = []
extra_options = {}
import sys
if "bdist_wininst" in sys.argv:
    install_script = "ae_post_install.py"
    extra_scripts.append(install_script)
    extra_scripts.append("AEViewer.ico")
    extra_options["bdist_wininst"] = {
        "install_script": install_script,
        "user_access_control": "auto",
    }
    
setup(name='ae',
      version='0.2',
      description='Python accoustic emission tools',
      author='Jozef Vesely',
      author_email='vesely@gjh.sk',
      url='http://github.com/jove1/ae',
      packages=['ae'],
      scripts=['AEViewer.py'] + extra_scripts,
      ext_modules=[
          Extension('ae.event_detector', 
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
              sources=['ae/event_detector.c'],
              ),
          ],
      requires=['numpy'],
      options=extra_options,
      )
