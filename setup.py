
from numpy.distutils.core import setup, Extension

setup(name='ae',
      version='0.1',
      description='Python accoustic emission tools',
      author='Jozef Vesely',
      author_email='vesely@gjh.sk',
      #url='',
      packages=['ae'],
      scripts=['viewer.py', 'events.py'],
      ext_modules=[
          Extension('ae.event_detector', sources = ['ae/event_detector.c']),
          ]
      )

