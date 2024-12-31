from sysconfig import get_path
from setuptools import setup, Extension

PATH_PREFIX = get_path('data')
module1 = Extension(f'greedy_builder',
                    extra_compile_args = ["-O3", "-std=c++20"],
                    define_macros = [('MAJOR_VERSION', '0'),
                                     ('MINOR_VERSION', '10')],
                    include_dirs = [f'{PATH_PREFIX}/include/',
                                    f'{PATH_PREFIX}/include/tbb',
                                    f'{PATH_PREFIX}/include/oneapi'],
                    library_dirs = [f'{PATH_PREFIX}/lib/'],
                    libraries = ['tbb'],
                    sources = ['pcatt/greedy_builder.cpp'])

setup(
    name="greedtok",
    version="0.1",
    description="Partition Cover Approach to Tokenization",
    author="JP Lim",
    author_email="jiapeng.lim.2021@phdcs.smu.edu.sg",
    license = "MIT",
    setup_requires=['pybind11', 'tbb-devel'],
    url = "https://github.com/PreferredAI/pcatt/",
    download_url = "https://github.com/PreferredAI/pcatt/archive/refs/tags/v0.10.tar.gz",
    ext_modules = [module1]
)
