#!/usr/bin/env python

#from distutils.core import setup, Extension
from setuptools import setup, Extension


import subprocess
try:
    a,b,c = subprocess.check_output(["git", "describe", "--tags", "--always"]).strip("v\n").split("-")
    __version__ = "{}.{}".format(a,b)
except subprocess.CalledProcessError:
    execfile('src/ae/version.py') # __version__
else:
    with open('src/ae/version.py','w') as f:
        f.write("__version__ = {!r}\n".format(__version__))


setup(
    name = 'ae',
    version = __version__,
    description = 'Python accoustic emission tools',
    author = 'Jozef Vesely',
    author_email = 'vesely@gjh.sk',
    url = 'http://github.com/jove1/ae',

    packages = ['ae'],
    package_dir = {'': 'src'},
    package_data = {
        'ae': ['*.doctest.rst'],
    },
    test_suite = 'ae.test.suite', # setuptools

    scripts = ['src/AEViewer.py', 'src/ae_post_install.py', 'src/AEViewer.ico'],
    #entry_points = {
    #   'console_scripts': ['helloworld = greatings.helloworld:main']
    #},
 
    ext_modules = [
        Extension('ae.event_detector',
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            sources=['src/ae/event_detector.c'],
        ),
    ],
    #requires = ['numpy'],
    setup_requires = ['numpy'],
    install_requires = ['numpy', 'scipy', 'matplotlib'],

    options = {
        "bdist_wininst": {
            "install_script": "ae_post_install.py",
            "user_access_control": "auto",
        }
    },
)
