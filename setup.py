
from numpy.distutils.core import setup

import ae.wfs
wfs_ext = ae.wfs.build_ext("ae")
wfs_ext.name = 'ae.' + wfs_ext.name

setup(name='ae',
      version='0.1',
      description='Python accoustic emission tools',
      author='Jozef Vesely',
      author_email='vesely@gjh.sk',
      #url='',
      packages=['ae'],
      scripts=['viewer.py', 'events.py'],
      ext_modules=[wfs_ext]
      )

