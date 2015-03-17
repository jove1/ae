


wininst_table = {
        "win32-2.5": "wininst-7.1.exe",
    "win-amd64-2.5": "wininst-8.0.exe",
        "win32-2.6": "wininst-9.0.exe",
    "win-amd64-2.6": "wininst-9.0-amd64.exe",
        "win32-2.7": "wininst-9.0.exe",
    "win-amd64-2.7": "wininst-9.0-amd64.exe",
}


msvct_table = {
        "2.5": "msvcr71",
        "2.6": "msvcr90",
        "2.7": "msvcr90",
}

def make_specs(compiler, msvcr, msvcrt_version=None, moldname=None):
    import subprocess
    specs = subprocess.Popen([compiler, "-dumpspecs"], stdout=subprocess.PIPE).communicate()[0].decode('ascii')

    import re
    newspecs = []
    m = re.search(r"(\*cpp:(?:\n|\r\n))(.*?)(\n\n|\r\n\r\n)", specs, re.DOTALL)
    if m and msvcrt_version is not None:
        a,b,c = m.groups()
        b += " -D__MSVCRT_VERSION__=0x%03i0" % (msvcrt_version,)
        newspecs.extend([a,b,c])
    m = re.search(r"(\*libgcc:(?:\n|\r\n))(.*?)(\n\n|\r\n\r\n)", specs, re.DOTALL)
    if m:
        a,b,c = m.groups()
        b = b.replace("msvcrt",msvcr)
        if moldname is not None:
            b = b.replace("moldname",moldname)
        newspecs.extend([a,b,c])
    return "".join(newspecs)

import sys, os, string

from distutils.command.build_ext import build_ext 
from distutils.command.build import build
from distutils.command.bdist_wininst import bdist_wininst
from distutils import log
from distutils.dir_util import remove_tree
from distutils.errors import DistutilsOptionError, DistutilsFileError

class cross_build(build):
    user_options = build.user_options + [
        ('cross-dir=', None,
         "Python headers and libraries for cross compilation"),
        ('cross-ver=', None,
         "version for cross compilation"),
        ('cross-compiler=', None,
         "compiler for cross compilation"),
    ]

    def initialize_options(self):
        build.initialize_options(self)
        self.cross_dir = None
        self.cross_ver = None
        self.cross_compiler = None

    def finalize_options(self):
        os_name = os.name
        sys_version = sys.version

        if self.plat_name: # allow plat_name override on non windows platforms
            os.name = "nt"

        if self.cross_ver: # version override
            sys.version = self.cross_ver

        build.finalize_options(self)

        sys.version = sys_version
        os.name = os_name

class cross_build_ext(build_ext):
    user_options = build_ext.user_options + [
        ('cross-dir=', None,
         "Python headers and libraries for cross compilation"),
        ('cross-ver=', None,
         "version for cross compilation"),
        ('cross-compiler=', None,
         "compiler for cross compilation"),
    ]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.cross_dir = None
        self.cross_ver = None
        self.cross_compiler = None
 
    def finalize_options(self):
        build_ext.finalize_options(self)
        self.set_undefined_options('build',
                ('cross_dir', 'cross_dir'),
                ('cross_ver', 'cross_ver'),
        )

        if self.cross_compiler is None:
            self.cross_compiler = { "win32": "i686-w64-mingw32-gcc",
                                    "win-amd64": "x86_64-w64-mingw32-gcc" }.get(self.plat_name)

    def run(self):
        if self.cross_dir is None:
            return build_ext.run(self)

        if not self.extensions:
            return

        if self.distribution.has_c_libraries():
            build_clib = self.get_finalized_command('build_clib')
            self.libraries.extend(build_clib.get_library_names() or [])
            self.library_dirs.append(build_clib.build_clib)

        from distutils.unixccompiler import UnixCCompiler

        self.compiler = UnixCCompiler(verbose=self.verbose,
                                      dry_run=self.dry_run,
                                      force=self.force)

        self.compiler.shared_lib_extension = ".pyd" # not used :(

        if self.libraries is not None:
            self.compiler.set_libraries(self.libraries)

        if self.library_dirs is not None:
            self.compiler.set_library_dirs(self.library_dirs)

       
        python_lib = "python"+self.cross_ver.replace(".","")

        import os.path, subprocess
        if not os.path.exists( os.path.join(self.cross_dir, python_lib+".def") ):
            log.info("making def for %s in %s", python_lib, self.cross_dir)
            subprocess.check_call(["gendef", python_lib+".dll"], cwd=self.cross_dir)
       
        python_lib_fname = self.compiler.static_lib_format % (python_lib, self.compiler.static_lib_extension )
        if not os.path.exists( os.path.join(self.cross_dir, python_lib_fname) ):
            log.info("making link library %s for %s in %s", python_lib_fname, python_lib, self.cross_dir)
            subprocess.check_call([self.cross_compiler.replace("gcc","dlltool"), 
                "--dllname", python_lib+".dll",
                "--def", python_lib+".def", 
                "--output-lib", python_lib_fname], cwd=self.cross_dir)

        specs_fname = os.path.join(self.cross_dir, "compiler.specs")
        if not os.path.exists(specs_fname):
            log.info("making compiler specs %s", specs_fname)
            msvcr = msvct_table.get(self.cross_ver)
            newspecs = make_specs(self.cross_compiler, msvcr, int(msvcr[-2:]))
            fh = open(specs_fname, "w")
            fh.write(newspecs)
            fh.close()

        self.compiler.set_executables(
            compiler_so="{} -specs={}".format(self.cross_compiler, specs_fname),
            linker_so="{} -specs={} -static-libgcc -shared".format(self.cross_compiler, specs_fname),
        )


        if self.plat_name == "win-amd64":
            self.compiler.define_macro("MS_WIN64")
        
        self.compiler.add_library(python_lib)
        self.compiler.add_library_dir(self.cross_dir)
        self.compiler.add_include_dir(os.path.join(self.cross_dir, 'include'))

        # Now actually compile and link everything.
        self.build_extensions()

    def get_ext_filename(self, ext_name):
        if self.cross_dir is None:
            return build_ext.get_ext_filename(self, ext_name)

        return build_ext.get_ext_filename(self, ext_name).replace(".so",".pyd")

class cross_bdist_wininst(bdist_wininst):
    user_options = build_ext.user_options + [
        ('wininst-exe=', None,
         "location of wininst.exe"),
        ('wininst-exe-path=', None,
         "location of wininst.exe")

    ]

    def initialize_options(self):
        bdist_wininst.initialize_options(self)
        self.wininst_exe = None
        self.wininst_exe_path = None

    def finalize_options(self):
        self.skip_build = True
        self.set_undefined_options('build', ('cross_ver', 'target_version'))
        bdist_wininst.finalize_options(self)


        if self.wininst_exe is None:
            import distutils.command.bdist_wininst
            directory = self.wininst_exe_path or os.path.dirname(distutils.command.bdist_wininst.__file__)
            try:
                wininst = wininst_table[self.plat_name+"-"+self.target_version]
            except KeyError:
                pass
            else:
                self.wininst_exe = os.path.join(directory, wininst)

    def run (self):
        if not self.skip_build:
            self.run_command('build')

        install = self.reinitialize_command('install', reinit_subcommands=1)
        install.root = self.bdist_dir
        install.skip_build = self.skip_build
        install.warn_dir = 0
        install.plat_name = self.plat_name

        install_lib = self.reinitialize_command('install_lib')
        # we do not want to include pyc or pyo files
        install_lib.compile = 0
        install_lib.optimize = 0

        if self.distribution.has_ext_modules():
            # If we are building an installer for a Python version other
            # than the one we are currently running, then we need to ensure
            # our build_lib reflects the other Python version rather than ours.
            # Note that for target_version!=sys.version, we must have skipped the
            # build step, so there is no issue with enforcing the build of this
            # version.
            target_version = self.target_version
            if not target_version:
                assert self.skip_build, "Should have already checked this"
                target_version = sys.version[0:3]
            plat_specifier = ".%s-%s" % (self.plat_name, target_version)
            build = self.get_finalized_command('build')
            build.build_lib = os.path.join(build.build_base,
                                           'lib' + plat_specifier)

        # Use a custom scheme for the zip-file, because we have to decide
        # at installation time which scheme to use.
        for key in ('purelib', 'platlib', 'headers', 'scripts', 'data'):
            value = string.upper(key)
            if key == 'headers':
                value = value + '/Include/$dist_name'
            setattr(install,
                    'install_' + key,
                    value)

        log.info("installing to %s", self.bdist_dir)
        install.ensure_finalized()

        # avoid warning of 'install_lib' about installing
        # into a directory not in sys.path
        sys.path.insert(0, os.path.join(self.bdist_dir, 'PURELIB'))

        install.run()

        del sys.path[0]

        # And make an archive relative to the root of the
        # pseudo-installation tree.
        from tempfile import mktemp
        archive_basename = mktemp()
        fullname = self.distribution.get_fullname()
        arcname = self.make_archive(archive_basename, "zip",
                                    root_dir=self.bdist_dir)
        # create an exe containing the zip-file
        self.create_exe(arcname, fullname, self.bitmap)
        if self.distribution.has_ext_modules():
            pyversion = self.target_version
        else:
            pyversion = 'any'
        self.distribution.dist_files.append(('bdist_wininst', pyversion,
                                             self.get_installer_filename(fullname)))
        # remove the zip-file again
        log.debug("removing temporary file '%s'", arcname)
        os.remove(arcname)

        if not self.keep_temp:
            remove_tree(self.bdist_dir, dry_run=self.dry_run)

    def get_exe_bytes (self):
        try:
            f = open(self.wininst_exe, "rb")
        except IOError, msg:
            raise DistutilsFileError, str(msg) + ', please install the python%s-dev package' % sys.version[:3]
        try:
            return f.read()
        finally:
            f.close()

cross_cmdclass = {
    'build': cross_build,
    'build_ext': cross_build_ext,
    'bdist_wininst': cross_bdist_wininst,
}
