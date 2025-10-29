from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import pybind11

__version__ = '0.1.0'

# Determine include directories
include_dirs = [
    pybind11.get_include(),
    "src/cpp",  # For vendored dlib headers
]

ext_modules = [
    Extension(
        'pyfhog._pyfhog',
        ['src/cpp/fhog_wrapper.cpp'],
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=['-std=c++14', '-O3'],  # Optimization
    ),
]

setup(
    name='pyfhog',
    version=__version__,
    author='John Wilson',
    author_email='',
    url='https://github.com/johnwilsoniv/pyfhog',
    description='Fast FHOG feature extraction compatible with OpenFace 2.2',
    long_description=open('README.md').read() if sys.path[0] else 'Fast FHOG feature extraction using dlib',
    long_description_content_type='text/markdown',
    packages=['pyfhog'],
    package_dir={'': 'src'},
    ext_modules=ext_modules,
    install_requires=['numpy>=1.20.0'],
    python_requires='>=3.8',
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: Boost Software License 1.0 (BSL-1.0)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
