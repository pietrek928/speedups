from Cython.Build import cythonize
from setuptools import setup, find_packages, Extension

optim_module = Extension(
    'optim',
    sources=['speedups/python.cc'],
    libraries=['boost_python38'],
    language='c++',
    extra_compile_args=["-std=c++17", "-O3"]
)

setup(
    name='speedups',
    packages=find_packages(),
    ext_package='speedups',
    ext_modules=cythonize([
        optim_module
    ])
)
