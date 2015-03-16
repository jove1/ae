    
from __future__ import print_function

import os, sys

def install():
    target = os.path.join(sys.prefix, 'python.exe')
    description = "AEViewer"
    workdir = "%HOMEDRIVE%%HOMEPATH%"
    arguments = '"{}"'.format(os.path.join(sys.prefix, "Scripts", "AEViewer.py"))
    iconpath =  os.path.join(sys.prefix, "Scripts", "AEViewer.ico")
    iconindex = 0
    
    for place in ['CSIDL_COMMON_PROGRAMS', 'CSIDL_COMMON_DESKTOPDIRECTORY']:
        linkdir = get_special_folder_path(place)
        filename = os.path.join(linkdir, "AEViewer.lnk")
        create_shortcut(target, description, filename, arguments, workdir, iconpath, iconindex)
        file_created(filename)

def remove():
    pass

if len(sys.argv) > 1:
    if sys.argv[1] == '-install':
        try:
            install()
        except OSError:
            print("Failed to create Start Menu items, try running the"
                  " installer as administrator.", file=sys.stderr)
    elif sys.argv[1] == '-remove':
        remove()
    else:
        print("Script was called with option %s" % sys.argv[1],
              file=sys.stderr)
