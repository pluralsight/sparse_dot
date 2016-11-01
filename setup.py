import os
import multiprocessing
from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

from distutils.command.sdist import sdist as _sdist

# Read the version number
with open("sparse_dot/_version.py") as f:
    exec(f.read())

extensions = [Extension('sparse_dot.cy_sparse_dot',
                        ['sparse_dot/cy_sparse_dot.pyx',
                         'sparse_dot/sparse_dot_core.c'],
                        include_dirs=['src'],
                        extra_compile_args=['-fPIC','-fopenmp'],
                        extra_link_args=['-fopenmp'])]
nthreads = multiprocessing.cpu_count()

class sdist(_sdist):
    def run(self):
        # Make sure the compiled Cython files in the distribution are up-to-date
        from Cython.Build import cythonize
        cythonize(extensions, nthreads=nthreads)
        _sdist.run(self)

setup(
    name='sparse_dot',
    version=__version__, # use the same version that's in _version.py
    author='David N. Mashburn',
    author_email='david.n.mashburn@gmail.com',
    packages=['sparse_dot'],
    scripts=[],
    license='LICENSE.txt',
    description='Highly efficient set statistics about many-to-many relationships',
    long_description=open('README.md').read(),
    install_requires=['numpy>=1.0',
                      'cython>=0.2', # this might need to be newer?
                      'np_utils>=0.4.6', # this is only used for testing currently
                     ],
    extras_require = {
                     },
    ext_modules=cythonize(extensions,
                          nthreads=nthreads),
    cmdclass={'sdist': sdist},
)
